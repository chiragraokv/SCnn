import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.linalg import hadamard
from PIL import Image

class HadamardCompression:
    def __init__(self, keep_ratio=0.1, size=32):
        self.keep_ratio = keep_ratio

        H = hadamard(size).astype(np.float32)
        self.H = H / np.sqrt(size)

    def hadamard2d(self, channel):
        return self.H @ channel @ self.H.T

    def inverse_hadamard2d(self, coeffs):
        return self.H.T @ coeffs @ self.H

    def compress_channel(self, channel):

        coeffs = self.hadamard2d(channel)

        flat = np.abs(coeffs).ravel()

        threshold = np.percentile(
            flat,
            100 * (1 - self.keep_ratio)
        )

        coeffs[np.abs(coeffs) < threshold] = 0

        reconstructed = self.inverse_hadamard2d(coeffs)

        return reconstructed

    def __call__(self, img):

        img = np.array(img).astype(np.float32)

        reconstructed = np.zeros_like(img)

        for c in range(3):
            reconstructed[:, :, c] = self.compress_channel(
                img[:, :, c]
            )

        reconstructed = np.clip(
            reconstructed,
            0,
            255
        ).astype(np.uint8)

        return Image.fromarray(reconstructed)
