"""Unit tests for HybridDefectDetector."""
import numpy as np
import cv2
import pytest
from vision_inspect.detector import HybridDefectDetector
from vision_inspect.config import DefectClass


def test_detector_normal_surface():
    detector = HybridDefectDetector()
    # Uniform smooth surface
    surface = np.full((256, 256, 3), 160, dtype=np.uint8)
    result = detector.inspect(surface)

    assert isinstance(result.is_defective, bool)
    assert result.latency_ms > 0
    assert result.preprocessed_img is not None
    assert result.anomaly_heatmap is not None


def test_detector_defective_surface():
    detector = HybridDefectDetector()
    surface = np.full((256, 256, 3), 160, dtype=np.uint8)
    # Add deep scratch
    cv2.line(surface, (50, 50), (220, 220), (20, 20, 20), 4)

    result = detector.inspect(surface)
    assert result.is_defective is True
    assert result.defect_count >= 1
    assert result.overall_severity in ["Minor", "Moderate", "Severe", "Critical"]
