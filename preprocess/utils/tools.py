# author: Kangran ZHAO
# email: kangranzhao@link.cuhk.edu.cn
# date: 2025-04-01
# description: audio-video dataset preprocessing tools

import cv2
import dlib
import librosa
import multiprocessing

import math
import numpy as np
import os
import shutil
import soundfile as sf
import subprocess

from skimage import transform as tf
from tqdm import tqdm

dlib_tool_path = os.path.join(os.path.dirname(__file__))
mean_face_landmarks = np.load(os.path.join(dlib_tool_path, '20words_mean_face.npy'))
STD_SIZE = (256, 256)
STABLE_POINTS = [33, 36, 39, 42, 45]
WINDOW_MARGIN = 12
FACE_SIZE = 256
MOUTH_SIZE = 96


def split_video_and_audio_multiprocess(data_list, num_proc):
    """
    Split video and audio from given list.
    Args:
        data_list: [list] each element is a dictionary containing attributes of this data.
        num_proc: [int] number of processors
    """
    if num_proc != 1:
        with multiprocessing.Pool(processes=num_proc) as pool:
            with tqdm(total=len(data_list), dynamic_ncols=True) as pbar:
                for _ in pool.imap_unordered(split_video_and_audio, data_list):
                    pbar.update(1)
    else:
        for data in tqdm(data_list, dynamic_ncols=True):
            split_video_and_audio(data)


def split_video_and_audio(data):
    """
    Split video and audio from one single *.mp4 file.
    Args:
        data: [dict] a dictionary containing attributes of this data
    """
    if os.path.exists(data['video_path']):
        os.remove(data['video_path'])
    if os.path.exists(data['audio_path']):
        os.remove(data['audio_path'])

    os.makedirs(os.path.dirname(data['video_path']), exist_ok=True)
    if data.get('original_path', None) is not None:
        try:
            cmd = ['ffmpeg', '-i', data['original_path'],
                   '-an', '-r', str(data['FPS']), '-c:v', 'libx264', '-crf', '23', data['video_path'],
                   '-vn', '-ar', str(data['sample_rate']), '-ac', '1', '-c:a', 'pcm_s16le', data['audio_path']]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Split audio and video failed: {data['output_path']}")
    elif data.get('original_video_path', None) is not None and data.get('original_audio_path', None) is not None:
        try:
            cmd = ['ffmpeg', '-i', data['original_video_path'], '-an', '-r', str(data['FPS']), '-c:v', 'libx264',
                   '-crf', '23', data['video_path']]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Adjusting video FPS failed: {data['output_path']}")

        try:
            cmd = ['ffmpeg', '-i', data['original_audio_path'], '-ar', str(data['sample_rate']), '-ac', '1',
                   '-c:a', 'pcm_s16le', data['audio_path']]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Adjusting audio sampling rate, format or channels failed: {data['output_path']}")
    else:
        raise NotImplementedError('Unsupported original data format')


def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    return np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                   int(round(center_x) - round(width)): int(round(center_x) + round(width))])


def video_and_audio_segmentation_multiprocess(data_list, num_proc):
    """
    Segment video and audio based on data_list.
    Args:
        data_list: [list] each element is a dictionary containing attributes of this data.
        num_proc: [int] number of processors
    """
    if num_proc != 1:
        with multiprocessing.Pool(processes=num_proc) as pool:
            with tqdm(total=len(data_list), dynamic_ncols=True) as pbar:
                for _ in pool.imap_unordered(video_and_audio_segmentation, data_list):
                    pbar.update(1)
    else:
        for data in tqdm(data_list, dynamic_ncols=True):
            video_and_audio_segmentation(data)


def video_and_audio_segmentation(data):
    """
    Segment video and corresponding audio if all consecutive frames in one required-length video segment have face detected.
    Args:
        data: [dict] data dictionary with file path stored.
    """

    def save_segment():
        # crop and save frames
        for i in range(len(frame_list)):
            smoothed_landmarks = np.mean(landmark_list[i:min(len(landmark_list), i + WINDOW_MARGIN)], axis=0)
            trans_frame, trans = warp_img(smoothed_landmarks[STABLE_POINTS, :],
                                          mean_face_landmarks[STABLE_POINTS, :],
                                          frame_list[i],
                                          STD_SIZE)
            landmark_list[i] = trans(landmark_list[i])
            frame_list[i] = cut_patch(trans_frame, landmark_list[i], FACE_SIZE // 2, FACE_SIZE // 2)

        # segment audio
        audio, sr = librosa.load(data['audio_path'], sr=data['sample_rate'])
        audio_cropped = audio[int((frame_idx + 1) / data['FPS'] * sr) - int(data['duration'] * data['sample_rate']):
                              int((frame_idx + 1) / data['FPS'] * sr)]
        if len(audio_cropped) < int(data['duration'] * data['sample_rate']):
            audio_cropped = np.pad(audio_cropped,
                                   (0, int(data['duration'] * data['sample_rate']) - len(audio_cropped)),
                                   mode='constant')

        # Write to disk
        # os.makedirs(os.path.join(save_root, current_folder), exist_ok=True)
        # if data['transcode']:
        #     np.savez_compressed(os.path.join(save_root, current_folder, 'data.npz'),
        #                         frames=np.stack(frame_list), landmarks=np.stack(landmark_list), audio=audio_cropped)
        # else:
        #     height, width = frame_list[0].shape[:2]
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #     out = cv2.VideoWriter(os.path.join(save_root, current_folder, 'frames.mp4'), fourcc, data['FPS'],
        #                           (width, height))
        #     for i in range(len(frame_list)):
        #         out.write(frame_list[i])
        #     out.release()
        #     np.save(os.path.join(save_root, current_folder, 'landmarks.npy'), np.stack(landmark_list))
        #     sf.write(os.path.join(save_root, current_folder, f'audio.wav'), audio_cropped, sr)

        os.makedirs(os.path.join(save_root.replace('preprocessed', 'transcoded'), current_folder), exist_ok=True)
        os.makedirs(os.path.join(save_root.replace('transcoded', 'preprocessed'), current_folder), exist_ok=True)
        np.savez_compressed(os.path.join(save_root.replace('preprocessed', 'transcoded'), current_folder, 'data.npz'),
                            frames=np.stack(frame_list), landmarks=np.stack(landmark_list), audio=audio_cropped)
        height, width = frame_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(save_root.replace('transcoded', 'preprocessed'), current_folder, 'frames.mp4'), fourcc, data['FPS'],
                              (width, height))
        for i in range(len(frame_list)):
            out.write(frame_list[i])
        out.release()
        np.save(os.path.join(save_root.replace('transcoded', 'preprocessed'), current_folder, 'landmarks.npy'), np.stack(landmark_list))
        sf.write(os.path.join(save_root.replace('transcoded', 'preprocessed'), current_folder, f'audio.wav'), audio_cropped, sr)
        frame_list.clear()
        landmark_list.clear()

    save_root = data['video_path'][:-4]
    if os.path.exists(save_root):  # if previous result exists, skip preprocessing of this data
        return

    total_frames = int(data['FPS'] * data['duration'])
    clip_cnt = 0
    current_folder = f'{0:04d}_{total_frames:04d}'
    frame_list, landmark_list = [], []
    face_detector = dlib.get_frontal_face_detector()
    cnn_detector = dlib.cnn_face_detection_model_v1(os.path.join(dlib_tool_path, 'mmod_human_face_detector.dat'))
    face_predictor = dlib.shape_predictor(os.path.join(dlib_tool_path, 'shape_predictor_68_face_landmarks.dat'))
    cap = cv2.VideoCapture(data['video_path'])
    if not cap.isOpened():
        print(f'Failed to open {data["video_path"]}')
        return

    # if partially manipulated, only consider fake segments
    if data.get('fake_segments', None) is None:
        target_frames = range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    else:
        target_frames = []
        for start_time, end_time in data['fake_segments']:
            start_frame = math.ceil(start_time * data['FPS'])
            end_frame = math.floor(end_time * data['FPS']) + 1
            if end_frame - start_frame < total_frames:
                start_frame = max(end_frame - total_frames, 0)
                end_frame = min(start_frame + total_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            target_frames.extend(list(range(start_frame, end_frame)))

        target_frames = list(set(target_frames))

    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        if frame_idx not in target_frames:
            current_folder = f'{frame_idx + 1:04d}_{frame_idx + total_frames + 1:04d}'
            frame_list.clear()
            landmark_list.clear()
            continue

        ret, frame = cap.read()
        if not ret:
            print(f'Failed to open {frame_idx}-th frame of {data["video_path"]}')
            current_folder = f'{frame_idx + 1:04d}_{frame_idx + total_frames + 1:04d}'
            frame_list.clear()
            landmark_list.clear()
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces) == 0:
            faces = cnn_detector(gray)
            faces = [d.rect for d in faces]
        # If no face detected or more than 1 face detected, discard previous results.
        # Todo: Multiple face case. How to match voice and speaker? How to track face? How to decide labels for each face?
        if len(faces) != 1:
            print(f'{len(faces)} faces detected in {frame_idx}-th frame of {data["video_path"]}')
            current_folder = f'{frame_idx + 1:04d}_{frame_idx + total_frames + 1:04d}'
            frame_list.clear()
            landmark_list.clear()
            continue

        landmarks = face_predictor(gray, faces[0])
        landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        frame_list.append(frame)
        landmark_list.append(landmarks)
        if len(frame_list) == total_frames:
            save_segment()
            current_folder = f'{frame_idx + 1:04d}_{frame_idx + total_frames + 1:04d}'
            clip_cnt += 1
            if clip_cnt == data['clip_amount']:
                break

    cap.release()
    os.remove(data['audio_path'])
    os.remove(data['video_path'])


def merge_audio_and_video_multiprocess(data_list, num_proc):
    """
    Merge video and audio for visualization using multi-process.
    Args:
        data_list: [list] each element is a dictionary containing attributes of this data.
        num_proc: [int] number of processors
    """
    with multiprocessing.Pool(processes=num_proc) as pool:
        with tqdm(total=len(data_list)) as pbar:
            for _ in pool.imap_unordered(merge_audio_and_video, data_list):
                pbar.update(1)


def merge_audio_and_video(data):
    """
    Merge video and audio for visualization
    Args:
        data: [dict] data dictionary with file path stored.
    """
    output_dir = os.path.dirname(data['audio_video_path'])
    temp_dir = os.path.join(output_dir, 'temp')
    if os.path.exists(data['audio_video_path']):
        return

    frames = np.load(data['video'])['frames']
    if len(frames) != int(data['FPS'] * data['duration']):
        raise AssertionError(
            f'Incorrect duration, got {len(frames)} frames, expect {int(data["FPS"] * data["duration"])} frames.')

    audio, sr = librosa.load(data['audio'], sr=None)
    if (len(audio) / sr - data['duration']) >= 0.01:
        raise AssertionError(f'Incorrect duration, got {len(audio) / sr:.2f}s, expect {data["duration"]}s')

    try:
        os.makedirs(temp_dir, exist_ok=True)
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, data['FPS'], (frames.shape[2], frames.shape[1]))
        for frame in frames:
            out.write(frame)
        out.release()

        cmd = ['ffmpeg', '-y', '-i', temp_video_path, '-i', data['audio'], '-c:v', 'copy', '-c:a', 'aac', '-strict',
               'experimental', '-shortest', data['audio_video_path']]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    finally:
        shutil.rmtree(temp_dir)
