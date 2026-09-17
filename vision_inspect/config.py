"""
Configuration module for VisionInspect-AI.
Defines defect categories, hyperparameters, visualization palettes, and configuration classes.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple


class DefectClass(Enum):
    NORMAL = "Normal"
    SCRATCH = "Scratch"
    CRACK = "Crack"
    PIT = "Pit"
    STAIN = "Stain"

    @classmethod
    def list_classes(cls) -> List[str]:
        return [c.value for c in cls]

    @classmethod
    def from_string(cls, name: str):
        for c in cls:
            if c.value.lower() == str(name).lower():
                return c
        return cls.NORMAL


# BGR Color palette for OpenCV rendering
DEFECT_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    DefectClass.NORMAL.value: (60, 180, 75),     # Green
    DefectClass.SCRATCH.value: (40, 40, 230),    # Red
    DefectClass.CRACK.value: (180, 50, 180),    # Magenta / Purple
    DefectClass.PIT.value: (30, 150, 240),      # Orange
    DefectClass.STAIN.value: (220, 160, 40),    # Cyan / Blue
}

# RGB Color palette for Matplotlib rendering
DEFECT_COLORS_RGB: Dict[str, Tuple[float, float, float]] = {
    DefectClass.NORMAL.value: (0.29, 0.70, 0.23),
    DefectClass.SCRATCH.value: (0.90, 0.16, 0.16),
    DefectClass.CRACK.value: (0.71, 0.20, 0.71),
    DefectClass.PIT.value: (0.94, 0.59, 0.12),
    DefectClass.STAIN.value: (0.16, 0.63, 0.86),
}


@dataclass
class PreprocessConfig:
    target_size: Tuple[int, int] = (256, 256)
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    bilateral_d: int = 7
    bilateral_sigma_color: float = 40.0
    bilateral_sigma_space: float = 40.0
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.2


@dataclass
class ClassicalEngineConfig:
    adaptive_block_size: int = 17
    adaptive_c: int = 4
    min_contour_area: int = 18
    max_contour_area: int = 15000
    tophat_kernel_size: int = 9
    blackhat_kernel_size: int = 9
    morph_open_kernel: int = 3
    morph_close_kernel: int = 5
    min_aspect_ratio_linear: float = 3.0
    circularity_pit_threshold: float = 0.55


@dataclass
class DeepEngineConfig:
    input_channels: int = 1
    image_size: int = 256
    latent_dim: int = 64
    learning_rate: float = 0.001
    batch_size: int = 8
    epochs: int = 10
    device: str = "cpu"
    anomaly_threshold: float = 0.040
    model_path: Path = Path("models/anomaly_autoencoder.pth")


@dataclass
class InspectionConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    classical: ClassicalEngineConfig = field(default_factory=ClassicalEngineConfig)
    deep: DeepEngineConfig = field(default_factory=DeepEngineConfig)
    fusion_deep_weight: float = 0.55
    fusion_classical_weight: float = 0.45
    min_confidence: float = 0.35
    severity_high_threshold: float = 0.70
    severity_medium_threshold: float = 0.40
    output_dir: Path = Path("data/results")
    sample_dir: Path = Path("data/samples")
