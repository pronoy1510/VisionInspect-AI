"""
Image preprocessing module for surface quality inspection.
Implements illumination correction, edge-preserving denoising, and gradient feature maps.
"""

from typing import Tuple, Union
import cv2
import numpy as np
from .config import PreprocessConfig


class ImagePreprocessor:
    """Standardizes, enhances, and extracts gradient representations from surface images."""

    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size
        )

    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Converts input image to single-channel 8-bit grayscale."""
        if len(img.shape) == 2:
            return img.copy()
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def resize(self, img: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Resizes image to target dimensions (width, height)."""
        target = target_size or self.config.target_size
        if (img.shape[1], img.shape[0]) == target:
            return img.copy()
        return cv2.resize(img, target, interpolation=cv2.INTER_AREA)

    def apply_clahe(self, gray_img: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization to balance lighting."""
        if gray_img.dtype != np.uint8:
            gray_img = (np.clip(gray_img, 0, 1) * 255).astype(np.uint8)
        return self.clahe.apply(gray_img)

    def denoise(self, gray_img: np.ndarray) -> np.ndarray:
        """Applies edge-preserving bilateral filtering followed by light Gaussian smoothing."""
        bilateral = cv2.bilateralFilter(
            gray_img,
            d=self.config.bilateral_d,
            sigmaColor=self.config.bilateral_sigma_color,
            sigmaSpace=self.config.bilateral_sigma_space
        )
        if self.config.gaussian_kernel_size > 1:
            k = self.config.gaussian_kernel_size
            if k % 2 == 0:
                k += 1
            return cv2.GaussianBlur(bilateral, (k, k), self.config.gaussian_sigma)
        return bilateral

    def compute_gradient_features(self, gray_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Sobel gradient magnitude and orientation.
        Returns:
            magnitude: float32 normalized [0, 1]
            orientation: float32 angles in radians [-pi, pi]
        """
        grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y)
        max_mag = np.max(magnitude)
        norm_mag = magnitude / (max_mag + 1e-7)
        return norm_mag.astype(np.float32), angle.astype(np.float32)

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Converts image to float32 normalized in range [0.0, 1.0]."""
        return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes end-to-end preprocessing pipeline.
        Returns:
            enhanced_uint8: (H, W) uint8 enhanced image
            normalized_float: (H, W) float32 in [0, 1]
            grad_magnitude: (H, W) float32 gradient magnitude in [0, 1]
        """
        resized = self.resize(img)
        gray = self.to_grayscale(resized)
        enhanced = self.apply_clahe(gray)
        denoised = self.denoise(enhanced)
        norm_float = self.normalize(denoised)
        grad_mag, _ = self.compute_gradient_features(denoised)
        return denoised, norm_float, grad_mag
