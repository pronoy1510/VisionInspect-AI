"""Unit tests for ClassicalDefectEngine."""
import numpy as np
import cv2
import pytest
from vision_inspect.classical_engine import ClassicalDefectEngine
from vision_inspect.config import DefectClass


def test_classical_engine_scratch_detection():
    engine = ClassicalDefectEngine()
    # Create blank surface with dark scratch
    surface = np.full((256, 256), 180, dtype=np.uint8)
    cv2.line(surface, (30, 100), (200, 110), 40, 3)

    candidates, mask = engine.detect(surface)
    assert len(candidates) >= 1
    # Check that scratch or crack is predicted
    types = [c.predicted_class for c in candidates]
    assert (DefectClass.SCRATCH in types) or (DefectClass.CRACK in types)


def test_classical_engine_pit_detection():
    engine = ClassicalDefectEngine()
    surface = np.full((256, 256), 180, dtype=np.uint8)
    # Draw circular dark pit
    cv2.circle(surface, (128, 128), 9, 30, -1)

    candidates, mask = engine.detect(surface)
    assert len(candidates) >= 1
    assert any(c.predicted_class == DefectClass.PIT for c in candidates)
