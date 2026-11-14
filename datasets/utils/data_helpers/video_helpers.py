"""Read sampled video/landmarks and normalize video inputs."""

import numpy as np

from pathlib import Path


def load_video_frames_fixed_stride(video_path, color_mode, sample_start_frame, sample_end_frame, sample_stride, mask_video):
    """
    Load video frames from a video file using a fixed stride sampling strategy.
    Args:
        video_path (str): Prepared video, e.g., 'xxx/8_297.mp4', covering source frames 8 through 296.
        color_mode (str): Color mode to load frames, options: ['RGB', 'Gray'].
        sample_start_frame (int): Start frame index to load from video.
        sample_end_frame (int): End frame index to load from video.
        sample_stride (int): Stride between frames.
        mask_video (bool): Return black frames instead of decoding the video.
    Returns:
        frames (np.ndarray): uint8 video in (T,H,W,C), range [0,255]; RGB for C=3 or grayscale for C=1.
    """
    from decord import VideoReader, cpu

    video_start_frame = int(str(Path(video_path).stem).split('_')[0])
    frame_ids = list(range(sample_start_frame - video_start_frame, sample_end_frame - video_start_frame, sample_stride))
    if mask_video:
        num_channels = 1 if color_mode == 'Gray' else 3
        frames = np.zeros((len(frame_ids), 256, 256, num_channels), dtype=np.uint8)
    else:
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = vr.get_batch(frame_ids).asnumpy()
        if color_mode == 'Gray':
            frames = np.dot(frames[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)[..., np.newaxis]

    return frames


def load_landmarks_fixed_stride(landmark_path, sample_start_frame, sample_end_frame, sample_stride):
    """
    Load video frames from a video file using a fixed stride sampling strategy.
    Args:
        landmark_path (str): Path to landmark file, e.g., 'xxx/8_297.npy' indicating the landmarks on 8th to 296-th
            frames after preprocessing.
        sample_start_frame (int): Start frame index to load from video.
        sample_end_frame (int): End frame index to load from video.
        sample_stride (int): Stride between frames.
    Returns:
        landmarks (np.ndarray) Landmark array with shape [T, 68, 2].
    """
    video_start_frame = int(str(Path(landmark_path).stem).split('_')[0])
    landmarks = np.load(landmark_path)
    return np.array(landmarks[sample_start_frame - video_start_frame : sample_end_frame - video_start_frame : sample_stride])


def video_normalization(video, mean=0, std=1):
    """
    Scale uint8 video and apply channel-wise normalization without changing color order.
    Args:
        video (np.ndarray): uint8 frames in (T,H,W,C), range [0,255]. RGB/BGR order is preserved;
            use statistics matching that order. Grayscale keeps C=1.
        mean (float or list): Scalar or per-channel mean on the [0,1] scale.
        std (float or list): Scalar or per-channel standard deviation on the [0,1] scale.
    Returns:
        normalized (np.ndarray): (video / 255 - mean) / std in the same layout. With mean=0/std=1,
            output is float32 in [0,1]; other statistics can produce values outside [0,1].
    """
    video = video.astype(np.float32) / 255.0
    if np.all(np.equal(mean, 0)) and np.all(np.equal(std, 1)):
        return video
    if np.ndim(mean):
        mean = np.asarray(mean).reshape(1, 1, 1, -1)
    if np.ndim(std):
        std = np.asarray(std).reshape(1, 1, 1, -1)
    return (video - mean) / std


def video_denormalization(video, mean=0, std=1):
    """
    Restore normalized video to uint8 pixels for display or saving.
    Args:
        video (np.ndarray): Floating-point video in (T,H,W,C), normalized by video_normalization.
        mean (float or list): The scalar or per-channel mean used during normalization.
        std (float or list): The scalar or per-channel standard deviation used during normalization.
    Returns:
        pixels (np.ndarray): uint8 video in (T,H,W,C), clipped/rounded to [0,255]. RGB/BGR order
            is preserved and grayscale keeps C=1. Defaults also convert float32 [0,1] to uint8.
    """
    if np.ndim(mean):
        mean = np.asarray(mean).reshape(1, 1, 1, -1)
    if np.ndim(std):
        std = np.asarray(std).reshape(1, 1, 1, -1)
    return np.round(np.clip(video * std + mean, 0, 1) * 255).astype(np.uint8)


def video_to_rgb_frames(video):
    """
    Convert display-ready channel-first video to RGB frames for Pillow or FFmpeg.
    Args:
        video (np.ndarray): uint8 pixels in (C,T,H,W), range [0,255]. C=3 requires RGB order;
            C=1 represents grayscale and must retain its channel dimension. BGR is not accepted.
    Returns:
        frames (np.ndarray): uint8 RGB frames in (T,H,W,3). Grayscale is repeated across RGB channels.
    """
    if video.ndim != 4:
        raise ValueError('Video must use (C,T,H,W); grayscale inputs must keep the C=1 dimension.')
    frames = video.transpose(1, 2, 3, 0)
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    return frames


if __name__ == '__main__':
    unit_test_params = {
        'video_path': 'data/examples/0_25.mp4',
        'color_mode': 'RGB',
        'sample_start_frame': 0,
        'sample_end_frame': 25,
        'sample_stride': 1,
        'mask_video': False,
    }
    frames = load_video_frames_fixed_stride(**unit_test_params)
    normalized = video_normalization(frames, mean=[0.5] * 3, std=[0.5] * 3)
    restored = video_denormalization(normalized, mean=[0.5] * 3, std=[0.5] * 3)
    print('Video frames:', frames.shape, frames.dtype)
    print('Normalization round trip:', np.array_equal(frames, restored))
