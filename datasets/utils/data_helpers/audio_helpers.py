"""Read audio windows, convert features and normalize spectrograms."""

import numpy as np


def load_audio_segment(audio_path, remove_dc_offset, start_time, end_time, audio_sampling_rate, temporal_shift, mask_audio):
    """
    Load audio_segments from an audio file.
    Args:
        audio_path (str): Path to audio file.
        remove_dc_offset (bool): Enable to remove DC offset, i.e., waveform - waveform.mean().
        start_time (float): Start time (s) to load from audio.
        end_time (float): End time (s) to load from audio.
        audio_sampling_rate (int): Audio sampling rate.
        temporal_shift (float): Temporal shift (s) of the audio.
        mask_audio: (bool) Enable when audio modality is masked, i.e., to load a silence audio segment.
    Returns:
        audio_segment (np.ndarray): float32 mono waveform (L,), normally in [-1,1]. Out-of-file
            positions are zero-padded. Optional DC removal can slightly exceed that range.
    """
    import librosa

    expected_len = int(round((end_time - start_time) * audio_sampling_rate))
    if mask_audio:
        audio_segment = np.zeros(expected_len, dtype=np.float32)
    else:
        shifted_start, shifted_end = start_time + temporal_shift, end_time + temporal_shift
        actual_load_start = max(0, shifted_start)
        actual_load_duration = shifted_end - actual_load_start
        if actual_load_duration <= 0:
            return np.zeros(expected_len, dtype=np.float32)

        audio, sr = librosa.load(audio_path, sr=audio_sampling_rate, offset=actual_load_start, duration=actual_load_duration)

        audio_segment = np.zeros(expected_len, dtype=np.float32)
        result_start = int(max(0, round((actual_load_start - shifted_start) * sr)))
        copy_len = min(len(audio), expected_len - result_start)
        if copy_len > 0:
            audio_segment[result_start : result_start + copy_len] = audio[:copy_len]

        if remove_dc_offset:
            audio_segment = audio_segment - np.mean(audio_segment)

    return audio_segment


def convert_audio_spectrogram(audio, sr, type, backend, params):
    """
    Convert audio to spectrogram.
    Args:
        audio (np.ndarray): float32 mono waveform (L,), normally in [-1,1].
        sr (int): Audio sampling rate.
        type (str) Type of audio_ spectrogram.
        backend (str) Backend to use.
        params (dict) Parameters sent to librosa, soundfile, python_speech_features, etc.
    Returns:
        spectrogram (np.ndarray): Floating-point features in (Ta,F); range depends on the chosen
            representation (e.g., magnitude, log energy or MFCC), not restricted to [0,1].
    """
    import librosa
    import python_speech_features
    import torch
    import torchaudio.compliance.kaldi as kaldi
    import torchaudio.transforms as T

    if backend == 'python_speech_features':
        audio = np.clip(audio, -1, 1)
        audio = (audio * 32767).astype(np.int16)
    elif backend in ['torchaudio', 'kaldi']:
        audio = torch.from_numpy(audio)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

    if type == 'STFT':
        if backend == 'librosa':
            spectrogram = np.abs(librosa.stft(audio, **params))
        elif backend == 'torchaudio':
            stft_transform = T.Spectrogram(**params)
            spectrogram = torch.abs(stft_transform(audio)).squeeze(0).numpy()
        else:
            raise NotImplementedError(f'STFT using {backend} not implemented.')
    elif type == 'Mel':
        if backend == 'librosa':
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, **params)
        elif backend == 'torchaudio':
            mel_transform = T.MelSpectrogram(sample_rate=sr, **params)
            spectrogram = mel_transform(audio).squeeze(0).numpy()
        elif backend == 'python_speech_features':
            spectrogram = python_speech_features.fbank(audio, samplerate=sr, **params)[0]
        else:
            raise NotImplementedError(f'Mel spectrogram using {backend} not implemented.')
    elif type == 'Log-Mel':
        if backend == 'librosa':
            params = params.copy()
            ref = params.pop('ref', np.max)
            spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, **params), ref=ref)
        elif backend == 'torchaudio':
            params = params.copy()
            top_db = params.pop('top_db', None)
            mel_transform = T.MelSpectrogram(sample_rate=sr, **params)
            amp_to_db = T.AmplitudeToDB(stype='power', top_db=top_db)
            spectrogram = amp_to_db(mel_transform(audio)).squeeze(0).numpy()
        elif backend == "python_speech_features":
            spectrogram = python_speech_features.logfbank(audio, samplerate=sr)
        elif backend == 'kaldi':
            spectrogram = kaldi.fbank(audio, sample_frequency=sr, **params).numpy()
        else:
            raise NotImplementedError('Log-Mel spectrogram using {backend} not implemented.')
    elif type == 'MFCC':
        if backend == 'librosa':
            spectrogram = librosa.feature.mfcc(y=audio, sr=sr, **params)
        elif backend == 'torchaudio':
            mfcc_transform = T.MFCC(sample_rate=sr, **params)
            spectrogram = mfcc_transform(audio).squeeze(0).numpy()
        elif backend == 'python_speech_features':
            spectrogram = python_speech_features.mfcc(audio, samplerate=sr, **params)
        else:
            raise NotImplementedError(f'Convert MFCC using {backend} not implemented.')
    else:
        raise NotImplementedError(f'{type} not implemented.')

    if backend in ['librosa', 'torchaudio']:
        spectrogram = spectrogram.T  # To ensure return a [T, F] ndarray

    return spectrogram


def spectrogram_normalization(spectrogram, mean=0, std=1):
    """
    Apply affine normalization to audio features without changing their layout.
    Args:
        spectrogram (np.ndarray): Floating-point audio features in (Ta,F).
        mean (float): Feature-normalization mean.
        std (float): Feature-normalization standard deviation.
    Returns:
        normalized (np.ndarray): (spectrogram - mean) / std in (Ta,F); no fixed value range is assumed.
    """
    if np.all(np.equal(mean, 0)) and np.all(np.equal(std, 1)):
        return spectrogram
    return (spectrogram - mean) / std


def spectrogram_denormalization(spectrogram, mean=0, std=1):
    """
    Restore audio features using their original normalization statistics.
    Args:
        spectrogram (np.ndarray): Normalized floating-point features in (Ta,F).
        mean (float): Mean originally used by spectrogram_normalization.
        std (float): Standard deviation originally used by spectrogram_normalization.
    Returns:
        restored (np.ndarray): spectrogram * std + mean in (Ta,F), preserving the feature scale.
    """
    if np.all(np.equal(mean, 0)) and np.all(np.equal(std, 1)):
        return spectrogram
    return spectrogram * std + mean


if __name__ == '__main__':
    unit_test_params = {
        'audio_path': 'data/examples/audio.wav',
        'remove_dc_offset': False,
        'start_time': 0,
        'end_time': 1,
        'audio_sampling_rate': 16000,
        'temporal_shift': 0,
        'mask_audio': False,
    }
    waveform = load_audio_segment(**unit_test_params)
    print('Audio waveform:', waveform.shape, waveform.dtype)
