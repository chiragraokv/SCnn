import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from torchvision.datasets import CIFAR10


dataset = CIFAR10(root="./data", train=False, download=True)

img, label = dataset[0]
img = np.array(img).astype(np.float32)

N = 32
H = hadamard(N).astype(np.float32)

# Normalize for orthogonality
H = H / np.sqrt(N)

def hadamard2d(channel):
    return H @ channel @ H.T

def inverse_hadamard2d(coeffs):
    return H.T @ coeffs @ H

def compress_channel(channel, keep_ratio=0.1):

    coeffs = hadamard2d(channel)

    flat = np.abs(coeffs).flatten()

    threshold = np.percentile(
        flat,
        100 * (1 - keep_ratio)
    )

    compressed = coeffs.copy()
    compressed[np.abs(compressed) < threshold] = 0

    reconstructed = inverse_hadamard2d(compressed)

    return reconstructed

reconstructed = np.zeros_like(img)

for c in range(3):
    reconstructed[:, :, c] = compress_channel(
        img[:, :, c],
        keep_ratio=0.1
    )

reconstructed = np.clip(reconstructed, 0, 255)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(img.astype(np.uint8))
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(reconstructed.astype(np.uint8))
plt.title("Hadamard 10% Coefficients")
plt.axis("off")

plt.show()