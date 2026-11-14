"""Flat RGB audio-video images and playable MP4 exports from display-ready NumPy inputs."""

import numpy as np
import os
import subprocess
import sys
import tempfile
import wave

from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps, font_manager
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.utils.data_helpers.video_helpers import video_to_rgb_frames


__all__ = ['present_audio_video', 'render_audio_video', 'render_spectrogram', 'render_waveform', 'save_audio_video']


def present_audio_video(
    video,
    audio,
    duration,
    output_path=None,
    *,
    frame_height=160,
    frame_gap=4,
    frame_border=2,
    audio_height=120,
    spectrogram_cmap="viridis",
    spectrogram_percentiles=(1.0, 99.0),
    waveform_color=(31, 78, 121),
    background_color="white",
    border_color=(25, 25, 25),
    margin=24,
    section_gap=18,
    show_labels=True,
    num_time_ticks=5,
    return_metadata=False,
):
    """
    Return one flat RGB image: sampled video frames above a time-aligned audio panel, not an MP4.
    Args:
        video (np.ndarray): uint8 pixels in (C,T,H,W), range [0,255]. Color is RGB with C=3;
            grayscale keeps C=1. Convert BGR, tensors or normalized floats before calling.
        audio (np.ndarray): Mono float32 waveform (L,) in [-1,1], or floating-point features (Ta,F).
            Features can be spectrograms/MFCCs with their own value scale; no fixed range is assumed.
        duration (float): Duration represented by both inputs, in seconds.
        output_path (str or Path or None): Optional image output path; None returns the image without saving.
        frame_height (int): Display height per video frame, in pixels.
        frame_gap (int): Horizontal gap between video frames.
        frame_border (int): Border width around video/audio panels.
        audio_height (int): Audio-panel height.
        spectrogram_cmap (str): Matplotlib colormap name; default viridis.
        spectrogram_percentiles (tuple): Lower/upper feature percentiles for display contrast.
        waveform_color (str or tuple): Waveform color, as a Pillow color or RGB tuple.
        background_color (str or tuple): Canvas background color.
        border_color (str or tuple): Border/text color.
        margin (int): Outer canvas margin.
        section_gap (int): Vertical gap between video and audio.
        show_labels (bool): Draw section labels and a shared time axis.
        num_time_ticks (int): Number of intervals on the time axis.
        return_metadata (bool): Also return frame/audio boxes used by Grad-CAM.
    Returns:
        image (PIL.Image.Image): One flat RGB image containing video and audio visualizations.
        metadata (dict): Layout/time coordinates, returned only when return_metadata=True.
    """
    # 1. Convert the explicitly channel-first video layout to RGB frames.
    frames = video_to_rgb_frames(video)
    if audio.ndim not in (1, 2):
        raise ValueError('Audio must be a mono waveform (L,) or feature matrix (Ta,F); remove any batch dimension.')
    audio_array = audio

    # 2. Calculate the canvas size based on all input video frames.
    input_frame_count = len(frames)
    frame_indices = np.arange(input_frame_count, dtype=np.int64)

    source_height, source_width = frames.shape[1:3]
    resized_frame_width = max(1, int(round(frame_height * source_width / source_height)))
    outer_frame_width = resized_frame_width + 2 * frame_border
    outer_frame_height = frame_height + 2 * frame_border
    timeline_width = len(frames) * outer_frame_width + max(0, len(frames) - 1) * frame_gap

    label_height = 26 if show_labels else 0
    tick_height = 28 if show_labels else 0
    top = margin + label_height
    audio_label_height = 24 if show_labels else 0
    audio_top = top + outer_frame_height + section_gap + audio_label_height
    canvas_width = timeline_width + 2 * margin
    canvas_height = audio_top + audio_height + 2 * frame_border + tick_height + margin

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=background_color)
    draw = ImageDraw.Draw(canvas)
    font_path = font_manager.findfont('DejaVu Sans')
    label_font = ImageFont.truetype(font_path, size=15)
    tick_font = ImageFont.truetype(font_path, size=12)

    # 3. Render all video frames from left to right.
    effective_fps = input_frame_count / float(duration)
    if show_labels:
        draw.text(
            (margin, margin),
            f"Video  |  {input_frame_count} sampled frames  |  {effective_fps:.2f} fps",
            fill=border_color,
            font=label_font,
        )

    frame_boxes = []
    x = margin
    for frame in frames:
        frame_image = Image.fromarray(frame, mode='RGB').resize((resized_frame_width, frame_height), Image.Resampling.LANCZOS)
        box = (x, top, x + outer_frame_width, top + outer_frame_height)
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=border_color)
        canvas.paste(frame_image, (x + frame_border, top + frame_border))
        frame_boxes.append(box)
        x += outer_frame_width + frame_gap

    # 4. Resize the complete audio representation to the video timeline width.
    audio_kind = "waveform" if audio_array.ndim == 1 else "spectrogram"
    if show_labels:
        audio_label = "Audio waveform" if audio_kind == "waveform" else "Audio spectrogram / features"
        draw.text((margin, audio_top - audio_label_height), audio_label, fill=border_color, font=label_font)

    audio_box = (
        margin,
        audio_top,
        margin + timeline_width,
        audio_top + audio_height + 2 * frame_border,
    )
    draw.rectangle((audio_box[0], audio_box[1], audio_box[2] - 1, audio_box[3] - 1), fill=border_color)
    if audio_kind == "waveform":
        audio_image = render_waveform(
            audio_array,
            width=timeline_width - 2 * frame_border,
            height=audio_height,
            line_color=waveform_color,
            background_color=background_color,
        )
    else:
        audio_image = render_spectrogram(
            audio_array,
            width=timeline_width - 2 * frame_border,
            height=audio_height,
            cmap=spectrogram_cmap,
            percentiles=spectrogram_percentiles,
        )
    canvas.paste(audio_image, (margin + frame_border, audio_top + frame_border))

    # Draw the shared time axis beneath the audio panel.
    if show_labels:
        left, right, axis_top = margin, margin + timeline_width, audio_box[3]
        draw.line((left, axis_top, right - 1, axis_top), fill=border_color, width=1)
        for index in range(num_time_ticks + 1):
            fraction = index / num_time_ticks
            tick_x = int(round(left + fraction * (right - left - 1)))
            draw.line((tick_x, axis_top, tick_x, axis_top + 5), fill=border_color, width=1)
            text = f'{fraction * duration:g}s'
            text_box = draw.textbbox((0, 0), text, font=tick_font)
            text_width = text_box[2] - text_box[0]
            text_x = min(max(left, tick_x - text_width // 2), right - text_width)
            draw.text((text_x, axis_top + 7), text, fill=border_color, font=tick_font)

    # 5. Record pixel positions for future Grad-CAM overlays.
    metadata = {
        "duration_seconds": float(duration),
        "input_frame_count": input_frame_count,
        "displayed_frame_indices": frame_indices.tolist(),
        "effective_sample_fps": effective_fps,
        "video_frame_boxes": frame_boxes,
        "video_frames": [
            {
                "input_index": int(index),
                "time_seconds": float(index) * float(duration) / input_frame_count,
                "box": box,
            }
            for index, box in zip(frame_indices, frame_boxes)
        ],
        "audio_box": audio_box,
        "audio_content_box": (
            audio_box[0] + frame_border,
            audio_box[1] + frame_border,
            audio_box[2] - frame_border,
            audio_box[3] - frame_border,
        ),
        "audio_kind": audio_kind,
        "spectrogram_layout": "TF" if audio_kind == "spectrogram" else None,
        "timeline_x_range": (margin, margin + timeline_width),
        "canvas_size": canvas.size,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)

    if return_metadata:
        return canvas, metadata
    return canvas


def render_spectrogram(spectrogram, width, height, cmap='viridis', percentiles=(1, 99)):
    """
    Render one feature matrix as a colored RGB image with time left-to-right and frequency bottom-to-top.
    Args:
        spectrogram (np.ndarray): Finite floating-point features in (Ta,F); values need not be in [0,1].
        width (int): Output width in pixels.
        height (int): Output height in pixels.
        cmap (str): Matplotlib colormap name, e.g., viridis, magma or gray.
        percentiles (tuple): Lower/upper feature percentiles for display contrast.
    Returns:
        image (PIL.Image.Image): RGB feature visualization, leaving input values unchanged.
    """
    # Transpose the feature axes and scale display contrast; constant features stay at zero.
    values = spectrogram.T.astype(np.float32, copy=False)
    low, high = np.percentile(values, percentiles)
    normalized = np.clip((values - low) / (high - low), 0, 1) if high > low else np.zeros_like(values)
    color_map = colormaps.get_cmap(cmap)
    rgb = np.round(color_map(np.flipud(normalized))[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR)


def render_waveform(waveform, width, height, line_color, background_color):
    """
    Render a one-dimensional waveform using a per-pixel min/max envelope.
    Args:
        waveform (np.ndarray): float32 mono waveform (L,), in [-1,1].
        width (int): Output image width in pixels.
        height (int): Output image height in pixels.
        line_color (str or tuple): Waveform color.
        background_color (str or tuple): Image background color.
    Returns:
        image (PIL.Image.Image): Rendered waveform image.
    """
    # Keep the fixed [-1,1] amplitude scale so gain changes remain visible.
    image = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(image)
    center = (height - 1) / 2.0
    draw.line((0, round(center), width - 1, round(center)), fill=(205, 205, 205), width=1)

    if len(waveform) <= width:
        x_values = np.linspace(0, width - 1, len(waveform))
        points = [(int(round(x)), int(round(center - value * center * 0.92))) for x, value in zip(x_values, waveform)]
        if len(points) == 1:
            draw.point(points[0], fill=line_color)
        else:
            draw.line(points, fill=line_color, width=1)
        return image

    # At display resolutions, a min/max envelope preserves short peaks much
    # better than interpolating or selecting one audio sample per pixel.
    boundaries = np.linspace(0, len(waveform), width + 1).astype(np.int64)
    for x in range(width):
        segment = waveform[boundaries[x] : boundaries[x + 1]]
        y_top = int(round(center - float(np.max(segment)) * center * 0.92))
        y_bottom = int(round(center - float(np.min(segment)) * center * 0.92))
        draw.line((x, y_top, x, y_bottom), fill=line_color, width=1)
    return image


def save_audio_video(video, audio, output_path, video_fps=25, audio_sample_rate=16000):
    """
    Save display-ready RGB video and its original waveform as one MP4 with an audio track.
    Args:
        video (np.ndarray): uint8 video in (C,T,H,W), [0,255], RGB for C=3 or grayscale for C=1.
        audio (np.ndarray): Finite float32 mono waveform (L,), in [-1,1], matching the video duration.
            Features such as MFCC/spectrograms are not audio tracks; provide the original waveform.
        output_path (str or Path): MP4 destination.
        video_fps (float): Playback frame rate, normally source FPS divided by the sampling stride.
        audio_sample_rate (int): Original waveform sample rate.
    Returns:
        None. H.264/AAC encoding is for inspection, not offline robustness generation.
    """
    frames = video_to_rgb_frames(video)
    if audio.ndim != 1:
        raise ValueError('MP4 export needs the original mono waveform (L,), not an audio feature matrix.')
    duration = len(frames) / video_fps
    if abs(len(audio) / audio_sample_rate - duration) > max(1 / video_fps, 1 / audio_sample_rate):
        raise ValueError('Audio/video durations differ; provide the matching waveform and sampled playback FPS.')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.presenter_', dir=output_path.parent) as work_dir:
        waveform_path = Path(work_dir) / 'audio.wav'
        temporary_output = Path(work_dir) / 'output.mp4'
        with wave.open(str(waveform_path), 'wb') as handle:
            handle.setparams((1, 2, int(audio_sample_rate), 0, 'NONE', 'not compressed'))
            handle.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
        _, height, width, _ = frames.shape
        command = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', f'{width}x{height}', '-r', str(video_fps), '-i', 'pipe:0',
            '-i', str(waveform_path), '-map', '0:v:0', '-map', '1:a:0',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-threads', '1', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '96k', '-shortest',
            '-movflags', '+faststart', str(temporary_output),
        ]
        try:
            subprocess.run(command, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(error.stderr.decode('utf-8', errors='replace').strip()) from error
        os.replace(temporary_output, output_path)


render_audio_video = present_audio_video


if __name__ == '__main__':
    # Self-contained manual example: synthetic frames, waveform and a colored spectrum.
    unit_test_params = {
        'output_directory': Path('outputs/presenter_examples'),
        'num_frames': 12,
        'video_fps': 12,
        'audio_sample_rate': 16000,
    }
    output_directory = unit_test_params['output_directory']
    output_directory.mkdir(parents=True, exist_ok=True)
    duration = unit_test_params['num_frames'] / unit_test_params['video_fps']
    x, y = np.meshgrid(np.arange(96), np.arange(64))
    frames = (
        np.stack(
            [
                np.stack(((x * 2 + index * 12) % 256, (y * 3) % 256, (x + y + index * 8) % 256), axis=-1)
                for index in range(unit_test_params['num_frames'])
            ]
        )
        .astype(np.uint8)
        .transpose(3, 0, 1, 2)
    )
    time = np.arange(int(duration * unit_test_params['audio_sample_rate'])) / unit_test_params['audio_sample_rate']
    waveform = (0.3 * np.sin(2 * np.pi * 220 * time) + 0.1 * np.sin(2 * np.pi * 660 * time)).astype(np.float32)
    spectrum = np.log1p(np.abs(np.fft.rfft(waveform[: len(waveform) // 256 * 256].reshape(-1, 256), axis=1)))
    present_audio_video(frames, waveform, duration, output_directory / 'waveform.png', frame_height=80)
    present_audio_video(frames, spectrum, duration, output_directory / 'spectrogram.png', frame_height=80)
    save_audio_video(
        frames,
        waveform,
        output_directory / 'audio_video.mp4',
        video_fps=unit_test_params['video_fps'],
        audio_sample_rate=unit_test_params['audio_sample_rate'],
    )
    print(f'Saved waveform/spectrogram presentations and an MP4 to {output_directory}')
