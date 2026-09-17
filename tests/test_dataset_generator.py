"""Unit tests for SyntheticSurfaceGenerator."""
from pathlib import Path
import numpy as np
import pytest
from vision_inspect.dataset_generator import SyntheticSurfaceGenerator
from vision_inspect.config import DefectClass


def test_surface_generator_types():
    gen = SyntheticSurfaceGenerator(seed=99)
    for tex in ["metal", "ceramic", "cast_iron"]:
        surf = gen.generate_base_texture(tex)
        assert surf.shape == (256, 256)
        assert surf.dtype == np.uint8

    img, mask, meta = gen.generate_sample(DefectClass.SCRATCH, texture="metal")
    assert img.shape == (256, 256, 3)
    assert mask.shape == (256, 256)
    assert meta["is_defective"] is True
    assert meta["type"] == DefectClass.SCRATCH.value


def test_batch_generator(tmp_path):
    gen = SyntheticSurfaceGenerator(seed=123)
    created = gen.generate_batch(count_per_class=1, output_dir=tmp_path)
    assert len(created) == 5  # 5 classes
    for p in created:
        assert p.exists()
        mask = p.parent / f"{p.stem}_mask.png"
        meta = p.parent / f"{p.stem}_meta.json"
        assert mask.exists()
        assert meta.exists()
