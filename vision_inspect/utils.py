"""
Utility module for logging, robust file I/O, serialization, and array validations.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
import numpy as np


def setup_logger(
    name: str = "VisionInspect",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Initializes and configures a standard logger with stdout and optional file logging."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def load_image(path: Union[str, Path], grayscale: bool = False) -> np.ndarray:
    """Loads an image with validation and format checks."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {path_obj.resolve()}")

    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path_obj), mode)
    if img is None:
        raise ValueError(f"Could not decode image at: {path_obj.resolve()}")
    return img


def save_image(path: Union[str, Path], img: np.ndarray) -> bool:
    """Saves an image to disk, ensuring directory structure exists."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path_obj), img)
    if not success:
        raise IOError(f"Failed to write image to: {path_obj.resolve()}")
    return True


def save_json(path: Union[str, Path], data: Dict[str, Any]) -> None:
    """Saves dictionary data to a formatted JSON file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads JSON data from file."""
    path_obj = Path(path)
    with open(path_obj, "r", encoding="utf-8") as f:
        return json.load(f)
