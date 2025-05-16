import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_centered_fft(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found!")

    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    log_magnitude = np.log(magnitude + 1e-10)

    plt.figure(figsize=(8, 6))
    plt.imshow(log_magnitude, cmap='viridis'), plt.title('Centered DFT Magnitude Spectrum')
    plt.colorbar()
    plt.savefig("fft.png")

plot_centered_fft("first_frame.png")
