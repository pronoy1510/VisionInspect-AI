"""
VisionInspect-AI: Industrial Surface Defect Detection and Quality Assessment Pipeline.
"""

__version__ = "1.0.0"
__author__ = "Pronoy"

from .config import InspectionConfig, DefectClass
from .detector import HybridDefectDetector, InspectionResult, DefectRegion
from .dataset_generator import SyntheticSurfaceGenerator
from .evaluator import ModelEvaluator
from .preprocessing import ImagePreprocessor
from .classical_engine import ClassicalDefectEngine
from .deep_engine import DeepAnomalyEngine

__all__ = [
    "InspectionConfig",
    "DefectClass",
    "HybridDefectDetector",
    "InspectionResult",
    "DefectRegion",
    "SyntheticSurfaceGenerator",
    "ModelEvaluator",
    "ImagePreprocessor",
    "ClassicalDefectEngine",
    "DeepAnomalyEngine",
]
