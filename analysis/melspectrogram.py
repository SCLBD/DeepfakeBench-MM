import librosa

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_and_save_logmel(audio_path, output_image_path="logmel.png", sr=22050, n_mels=128):
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)

    plt.figure(figsize=(8, 6))
    librosa.display.specshow(log_mel, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram')
    plt.tight_layout()

    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()

# 使用示例
generate_and_save_logmel("audio.wav", "mfcc_visualization.png")