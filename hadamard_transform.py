import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from PIL import Image

# Load image
img = Image.open("/home/laserhammer/Downloads/WhatsApp Image 2026-05-10 at 23.36.51.jpeg").convert("RGB")

# Resize because Hadamard matrix size must match image dimensions
img = img.resize((256, 256))

img = np.array(img).astype(np.float32)

N = 256
H = hadamard(N).astype(np.float32) / np.sqrt(N)

def hadamard2d(channel):
    return H @ channel @ H.T

def inverse_hadamard2d(coeffs):
    return H.T @ coeffs @ H

def compress_channel(channel, keep_ratio=0.1):
    coeffs = hadamard2d(channel)

    flat = np.abs(coeffs).ravel()

    threshold = np.percentile(
        flat,
        100 * (1 - keep_ratio)
    )

    compressed = coeffs.copy()
    compressed[np.abs(compressed) < threshold] = 0

    return inverse_hadamard2d(compressed)

reconstructed = np.zeros_like(img)

ratios = [0.5, 0.1, 0.05, 0.01]

plt.figure(figsize=(12,8))

for i, ratio in enumerate(ratios):
    reconstructed = np.zeros_like(img)

    for c in range(3):
        reconstructed[:, :, c] = compress_channel(
            img[:, :, c],
            keep_ratio=ratio
        )

    reconstructed = np.clip(reconstructed, 0, 255)

    plt.subplot(2, 2, i+1)
    plt.imshow(reconstructed.astype(np.uint8))
    plt.title(f"Keep {ratio*100:.0f}%")
    plt.axis("off")

plt.show()