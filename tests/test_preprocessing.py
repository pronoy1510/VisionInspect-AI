"""Unit tests for ImagePreprocessor."""
import numpy as np
import pytest
from vision_inspect.preprocessing import ImagePreprocessor
from vision_inspect.config import PreprocessConfig


def test_to_grayscale_and_resize():
    pre = ImagePreprocessor()
    dummy_color = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    gray = pre.to_grayscale(dummy_color)
    assert len(gray.shape) == 2
    assert gray.shape == (300, 400)

    resized = pre.resize(dummy_color, target_size=(256, 256))
    assert resized.shape == (256, 256, 3)


def test_preprocess_pipeline():
    pre = ImagePreprocessor()
    dummy = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
    enhanced, norm, grad = pre.preprocess(dummy)

    assert enhanced.shape == (256, 256)
    assert enhanced.dtype == np.uint8
    assert norm.shape == (256, 256)
    assert norm.dtype == np.float32
    assert 0.0 <= norm.min() <= norm.max() <= 1.0
    assert grad.shape == (256, 256)
    assert 0.0 <= grad.min() <= grad.max() <= 1.0
