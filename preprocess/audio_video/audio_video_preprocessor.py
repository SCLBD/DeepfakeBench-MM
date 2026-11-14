import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import dlib
import multiprocessing
import numpy as np
import shutil
import subprocess
import sys

from pathlib import Path
from skimage import transform as tf
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocess.abstract_preprocessor import AbstractPreprocessor


cv2.setNumThreads(0)


cnn_lock = multiprocessing.Lock()
dlib_tool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dlib_tools')


SKIP_FRAME = 2
KEY_POINTS = [33, 36, 39, 42, 45]
CROP_SIZE = 256
WINDOW_SIZE = 12
LANDMARK_TEMPLATE = np.load(os.path.join(dlib_tool_path, '20words_mean_face.npy'))


class AudioVideoPreprocessor(AbstractPreprocessor):
    def __init__(self, dataset_name, split):
        self.fps = 25
        self.audio_sampling_rate = 16000
        super(AudioVideoPreprocessor, self).__init__(dataset_name, 'audio-video', split)

    def preprocess_all_data(self, num_proc):
        """
        Run data preprocessing if multiple processor provided.
        Args:
            num_proc: [int] number of processors
        """
        if num_proc != 1:
            with multiprocessing.Pool(processes=num_proc) as pool:
                with tqdm(total=len(self.data_list), desc=f'Preprocssing {self.dataset_name}', dynamic_ncols=True) as pbar:
                    for _ in pool.imap_unordered(preprocess_one_data, self.data_list):
                        pbar.update(1)
        else:
            for data in tqdm(self.data_list, desc=f'Preprocssing {self.dataset_name}', dynamic_ncols=True):
                preprocess_one_data(data)

    def assemble_data_split(self, data_split, data_list):
        """
        Assemble specified data split into list of dictionaries.
        Args:
            data_split: [str] data split name, options: ['train', 'val', 'test']
            data_list: [list] list of prefetched data information list.
        Returns:
             json_list: [list] list of dictionaries, which contains the final data information.
        """
        json_list = []
        for data in tqdm(data_list, desc=f'Assembling {data_split}', dynamic_ncols=True):
            if not os.path.exists(data['output_path'] + '.done'):  # fail to preprocess
                continue

            segment_list = list(Path(data['output_path']).glob('*.mp4'))
            assert len(segment_list) == 1
            # start_frame, end_frame = str(segment_list[0].stem).split('_')
            # start_frame, end_frame = int(start_frame), int(end_frame)
            # data['start_frame'], data['end_frame'] = start_frame, end_frame
            data['original_path'] = str(Path(data['original_path']).relative_to(self.path['original_dir']))
            data['path'] = str(segment_list[0].relative_to(Path(self.path['preprocessed_dir']) / 'audio-video'))
            del data['output_path']
            json_list.append(data)

        return json_list

def remove_cache(output_path):
    """
    Remove cached files.
    Args:
        output_path: (str) Path to output folder. Possible cache: {output_path}.mp4, {output_path}/*
    """
    if os.path.exists(output_path + '.wav'):
        os.remove(output_path + '.wav')
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

def warp_img(src, dst, img):
    """
    Use provided 5 key points to calculate affine matrix and apply transform.
    Args:
        src (np.array): 5 key points in provided frame.
        dst (np.array): 5 key points in template.
        img (np.array): Image to warp.
    Returns:
        warped_img (np.array): Warped image.
        tform (np.array): Transform matrix.
    """
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=(256, 256))  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform

def cut_patch(img, landmarks, height, width):
    """
    Cut patch on the aligned image, centered at the mean of landmarks.
    Args:
        img (np.array): Image to cut patch on.
        landmarks (np.array): Selected landmarks on the aligned image. Specifically, mouth landmarks.
        height (int): Height of the patch. In AVHuBERT, use 48 for mouth region cut.
        width (int): Width of the patch. In AVHuBERT, use 48 for mouth region cut.
    """
    center_x, center_y = np.mean(landmarks, axis=0)
    center_x = max(width, center_x)
    center_x = min(img.shape[1] - width, center_x)
    center_y = max(height, center_y)
    center_y = min(img.shape[0] - height, center_y)
    return np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                       int(round(center_x) - round(width)): int(round(center_x) + round(width))])


def preprocess_one_data(data, silence=True):
    """
    Preprocess one audio-video data.
    Args:
        data: (dict) A dictionary containing information of this audio_video data, requiring keys and values:
            1) 'original_path': data original path, if original data is provided in the format of *.mp4 containing
                both video and audio.
            2) 'original_path_audio' and 'original_path_video': alternative of 1), when original data is provided in
                the format of one audio and one video file.
            3) 'output_path': output path, e.g., original path is 'xxx/00001.mp4', output path is 'yyy/00001'. The
                temporary file will be saved at 'yyy/00001/' and the file output should be 'yyy/00001.wav',
                'yyy/00001/{start_frame}_{end_frame}.mp4, and 'yyy/00001/{start_frame}_{end_frame}.npy'.
            4) 'sampling_rate': audio sampling rate, leave it as None to use the default value 16,000.
            5) 'FPS': video FPS, leave it as None to use the default value 25.
        silence: (bool, default=True) Whether to hide progress in console.
    """
    # [Step 0] if *.done or *.abort file exists, skip this file to save time.
    if os.path.exists(data['output_path'] + '.done') or os.path.exists(data['output_path'] + '.abort'):
        return

    # [Step 1] Separate audio and video stream. Adjust audio sampling rate and video frame rate.
    remove_cache(data['output_path'])
    if not silence:
        print('[Step 1] Separate audio and video stream. Adjust audio sampling rate and video frame rate.')
    os.makedirs(data['output_path'], exist_ok=True)
    if data.get('original_path'):  # one mp4 file containing audio and video
        cmd = [
            'ffmpeg', '-threads', '1', '-i', data['original_path'],
            '-an', '-r', str(data.get('FPS', 25)),
            '-c:v', 'libx264', '-crf', '23', os.path.join(data['output_path'], 'temp_video.mp4'),
            '-vn', '-ar', str(data.get('sample_rate', 16000)),
            '-ac', '1', '-c:a', 'pcm_s16le', data['output_path'] + '.wav'
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f'Step 1 failed: {data["original_path"]}. Expected operation: split audio and video stream, '
                  f'adjust audio sampling rate and video frame rate.')
            remove_cache(data['output_path'])
            return
    elif data.get('original_path_audio') and data.get('original_path_video'):  # one audio and one video file
        cmd_video = [
            'ffmpeg', '-i', data['original_path_video'],
            '-an', '-r', str(data.get('FPS', 25)),
            '-c:v', 'libx264', '-crf', '23', os.path.join(data['output_path'], 'temp_video.mp4'),
        ]
        cmd_audio = [
            'ffmpeg', '-i', data['original_path_audio'],
            '-ar', str(data.get('sample_rate', 16000)),
            '-ac', '1', '-c:a', 'pcm_s16le', data['output_path'] + '.wav'
        ]
        try:
            subprocess.run(cmd_video, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f'Step 1 failed: {data["original_path_audio"]} and {data["original_path_video"]}. Expected '
                  f'operation: adjust audio sampling rate and video frame rate.')
            remove_cache(data['output_path'])
            return
    else:
        print('Provide either mp4 file path containing both video and audio or two file paths containing audio and '
              'video respectively.')
        remove_cache(data['output_path'])
        return

    # [Step 2] Extract landmarks for each frame. During extraction, we check whether any frame contains no face or more
    # than 1 face. If either of this exception happens, we consider this video have no face detected to avoid multiple
    # face cases. Extracted landmarks will be saved as 'yyy/00001/{start_frame}_{end_frame}.npy' in the format of ndarray
    # in shape [(end_frame - start_frame), 68, 2]. Only the longest sequences will be saved. If the longest sequence is
    # shorter than 1 second, we discard this data.
    if not silence:
        print('[Step 2] Extract landmarks for each frame.')
    face_detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(os.path.join(dlib_tool_path, 'mmod_human_face_detector.dat'))
    shape_predictor = dlib.shape_predictor(os.path.join(dlib_tool_path, 'shape_predictor_68_face_landmarks.dat'))

    cap = cv2.VideoCapture(os.path.join(data['output_path'], 'temp_video.mp4'))
    if not cap.isOpened():
        print(f'Fail to open temporary video: {os.path.join(data["output_path"], "temp_video.mp4")}')
        remove_cache(data['output_path'])
        return

    landmark_list = []
    start_frame = SKIP_FRAME
    for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), dynamic_ncols=True, disable=silence):
        ret, frame = cap.read()
        if not ret:
            print(f'Fail to read frame: {frame_idx} of {os.path.join(data["output_path"], "temp_video.mp4")}')
            remove_cache(data['output_path'])
            return

        if frame_idx < SKIP_FRAME:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with cnn_lock:
            face_rects = face_detector(gray, 1)
            if len(face_rects) == 0:
                face_rects = cnn_detector(gray)
                face_rects = [_.rect for _ in face_rects]

        if len(face_rects) != 1:  # collected previous frames if more than 1s frames collected
            if frame_idx - start_frame >= int(data.get('FPS', 25)):
                np.save(os.path.join(data['output_path'], f'{start_frame}_{frame_idx}.npy'), np.stack(landmark_list))

            start_frame = frame_idx + 1
            landmark_list = []
        else:  # detect the landmark of the only face
            landmark = shape_predictor(gray, face_rects[0])
            landmark = np.array([[landmark.part(i).x, landmark.part(i).y] for i in range(68)])
            landmark_list.append(landmark)

    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame >= int(data.get('FPS', 25)):    # collect last time
        np.save(os.path.join(data['output_path'], f'{start_frame}_{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}.npy'),
                np.stack(landmark_list))
    cap.release()

    # find the longest landmark sequence
    longest_sequence = None
    longest_sequence_length = 0
    for file in list(Path(data['output_path']).glob('*.npy')):
        sequence_start, sequence_end = str(file.stem).split('_')
        sequence_start, sequence_end = int(sequence_start), int(sequence_end)
        if sequence_end - sequence_start > longest_sequence_length:
            longest_sequence_length = sequence_end - sequence_start
            longest_sequence = file

    if longest_sequence is None:
        print(f'Step 2 Failed. No consecutive 1-second-long frame have 1 face detected. Expected output path: '
              f'{data["output_path"]}')
        remove_cache(data['output_path'])
        Path(data['output_path'] + '.abort').touch()
        return
    else:  # remove landmark sequences which is not the longest one
        for file in list(Path(data['output_path']).glob('*.npy')):
            if file != longest_sequence:
                os.remove(file)

    # [Step 3] Align face with landmark template
    if not silence:
        print('[Step 3] Align face with landmark template')
    sequence_start, sequence_end = str(longest_sequence.stem).split('_')
    sequence_start, sequence_end = int(sequence_start), int(sequence_end)
    landmark_sequence = np.load(str(longest_sequence))
    cap = cv2.VideoCapture(os.path.join(data['output_path'], 'temp_video.mp4'))
    if not cap.isOpened():
        print(f'Fail to open temporary video: {os.path.join(data["output_path"], "temp_video.mp4")}')
        remove_cache(data['output_path'])
        return

    landmark_list = []  # to collect transformed landmark
    for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), dynamic_ncols=True, disable=silence):
        ret, frame = cap.read()
        if not ret:
            print(f'Fail to read frame: {frame_idx} of {os.path.join(data["output_path"], "temp_video.mp4")}')
            remove_cache(data['output_path'])
            return

        if frame_idx < sequence_start or frame_idx >= sequence_end:
            continue

        # post-period average window, apply similarity transformation, and cut patch
        smoothed_landmark = np.mean(
            landmark_sequence[frame_idx - sequence_start:
                              min(frame_idx + WINDOW_SIZE, sequence_end) - sequence_start], axis=0
        )
        frame, tform = warp_img(smoothed_landmark[KEY_POINTS, :], LANDMARK_TEMPLATE[KEY_POINTS, :], frame)
        landmark_current_frame = tform(landmark_sequence[frame_idx - sequence_start])
        landmark_list.append(landmark_current_frame)
        cropped_frame = cut_patch(frame, landmark_current_frame[48:68], CROP_SIZE // 2, CROP_SIZE // 2)
        cv2.imwrite(os.path.join(data['output_path'], f'{frame_idx}.png'), cropped_frame)

    cap.release()
    np.save(os.path.join(data['output_path'], f'{sequence_start}_{sequence_end}.npy'), np.stack(landmark_list))

    # collect frames into mp4 video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(data['output_path'], f'{sequence_start}_{sequence_end}.mp4'), fourcc,
                          data.get('FPS', 25), (CROP_SIZE, CROP_SIZE))
    for i in range(sequence_start, sequence_end):
        img = cv2.imread(os.path.join(data['output_path'], f'{i}.png'))
        out.write(img)
        os.remove(os.path.join(data['output_path'], f'{i}.png'))
    out.release()

    # remove cache and add success label
    os.remove(os.path.join(data['output_path'], 'temp_video.mp4'))
    Path(data['output_path'] + '.done').touch() # create an empty *.done file to mark this data has been preprocessed


if __name__ == '__main__':
    preprocessed_data = {
        'original_path': '/data/original/FakeAVCeleb_v1.2/RealVideo-RealAudio/African/men/id00076/00109.mp4',
        'output_path': '/workspace/temp_test',
    }
    preprocess_one_data(preprocessed_data, silence=False)