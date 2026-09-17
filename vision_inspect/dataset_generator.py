"""
Procedural Industrial Surface and Defect Dataset Generator.
Generates realistic base textures (brushed metal, ceramic, cast iron) and injects calibrated defects.
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from .config import DefectClass
from .utils import save_image, save_json


class SyntheticSurfaceGenerator:
    """Generates synthetic industrial surface images with ground-truth masks and annotations."""

    def __init__(self, width: int = 256, height: int = 256, seed: Optional[int] = 42):
        self.width = width
        self.height = height
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_base_texture(self, texture_type: str = "metal") -> np.ndarray:
        """Generates realistic base surface texture."""
        h, w = self.height, self.width
        if texture_type == "metal":
            # Brushed metal: horizontal streaks + gentle lighting gradient
            base = np.full((h, w), 160, dtype=np.float32)
            noise = np.random.normal(0, 12, (h, w)).astype(np.float32)
            # Blur strongly horizontally to make brushed streaks
            streaks = cv2.GaussianBlur(noise, (15, 1), 0)
            # Subtle lighting gradient
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            xx, yy = np.meshgrid(x, y)
            vignette = -20 * (xx ** 2 + yy ** 2)
            surface = base + streaks + vignette
        elif texture_type == "cast_iron":
            # Rough cast iron: isotropic granular texture
            base = np.full((h, w), 110, dtype=np.float32)
            speckle = np.random.normal(0, 18, (h, w)).astype(np.float32)
            surface = base + speckle
        else:
            # Ceramic: smooth with very light micro-texture
            base = np.full((h, w), 200, dtype=np.float32)
            fine_noise = np.random.normal(0, 4, (h, w)).astype(np.float32)
            surface = base + fine_noise

        return np.clip(surface, 20, 245).astype(np.uint8)

    def inject_scratch(self, surface: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Injects a high-aspect-ratio linear/curved scratch with specular highlights."""
        img = surface.copy()
        mask = np.zeros_like(surface, dtype=np.uint8)

        x1 = random.randint(20, self.width - 50)
        y1 = random.randint(20, self.height - 50)
        length = random.randint(50, 140)
        angle = random.uniform(0, 2 * math.pi)

        x2 = int(np.clip(x1 + length * math.cos(angle), 5, self.width - 5))
        y2 = int(np.clip(y1 + length * math.sin(angle), 5, self.height - 5))
        thickness = random.randint(2, 4)

        # Draw dark groove
        cv2.line(img, (x1, y1), (x2, y2), int(np.mean(surface) - 65), thickness)
        # Draw specular adjacent highlight
        ox = 1 if math.sin(angle) > 0 else -1
        oy = 1 if math.cos(angle) > 0 else -1
        cv2.line(img, (x1 + ox, y1 + oy), (x2 + ox, y2 + oy), min(255, int(np.mean(surface) + 45)), 1)
        # Mask
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness + 2)

        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        bbox = (max(0, x_min - 4), max(0, y_min - 4), min(self.width, x_max - x_min + 8), min(self.height, y_max - y_min + 8))

        return img, mask, {"type": DefectClass.SCRATCH.value, "bbox": bbox}

    def inject_crack(self, surface: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Injects a jagged branching crack using random walk."""
        img = surface.copy()
        mask = np.zeros_like(surface, dtype=np.uint8)

        cx = random.randint(40, self.width - 60)
        cy = random.randint(40, self.height - 60)
        points = [(cx, cy)]

        steps = random.randint(35, 75)
        angle = random.uniform(0, 2 * math.pi)
        pts_x = [cx]
        pts_y = [cy]

        curr_x, curr_y = cx, cy
        for _ in range(steps):
            angle += random.uniform(-0.55, 0.55)
            step_size = random.uniform(2.0, 4.0)
            curr_x = int(np.clip(curr_x + step_size * math.cos(angle), 5, self.width - 5))
            curr_y = int(np.clip(curr_y + step_size * math.sin(angle), 5, self.height - 5))
            points.append((curr_x, curr_y))
            pts_x.append(curr_x)
            pts_y.append(curr_y)

        # Draw crack path with varying thickness
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            cv2.line(img, p1, p2, 25, 2)
            cv2.line(mask, p1, p2, 255, 3)

        x_min, x_max = min(pts_x), max(pts_x)
        y_min, y_max = min(pts_y), max(pts_y)
        bbox = (max(0, x_min - 4), max(0, y_min - 4), min(self.width, x_max - x_min + 8), min(self.height, y_max - y_min + 8))

        return img, mask, {"type": DefectClass.CRACK.value, "bbox": bbox}

    def inject_pit(self, surface: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Injects a circular void/pit with shadow rim."""
        img = surface.copy()
        mask = np.zeros_like(surface, dtype=np.uint8)

        cx = random.randint(30, self.width - 30)
        cy = random.randint(30, self.height - 30)
        radius = random.randint(5, 14)

        # Crater core
        cv2.circle(img, (cx, cy), radius, int(np.mean(surface) - 80), -1)
        # Specular rim on upper-left
        cv2.ellipse(img, (cx, cy), (radius + 2, radius + 2), 0, 180, 270, min(255, int(np.mean(surface) + 50)), 1)
        # Mask
        cv2.circle(mask, (cx, cy), radius + 2, 255, -1)

        bbox = (max(0, cx - radius - 3), max(0, cy - radius - 3), 2 * radius + 6, 2 * radius + 6)
        return img, mask, {"type": DefectClass.PIT.value, "bbox": bbox}

    def inject_stain(self, surface: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Injects a diffuse organic or oil stain blotch."""
        img = surface.copy().astype(np.float32)
        mask = np.zeros_like(surface, dtype=np.uint8)

        cx = random.randint(40, self.width - 40)
        cy = random.randint(40, self.height - 40)
        rx = random.randint(15, 35)
        ry = random.randint(12, 28)
        rot = random.randint(0, 180)

        temp = np.zeros_like(surface, dtype=np.float32)
        cv2.ellipse(temp, (cx, cy), (rx, ry), rot, 0, 360, 1.0, -1)
        temp = cv2.GaussianBlur(temp, (21, 21), 6.0)

        # Darken the stained region
        img = img - (temp * 50.0)
        img = np.clip(img, 0, 255).astype(np.uint8)

        mask[temp > 0.3] = 255
        x, y, w, h = cv2.boundingRect((temp > 0.3).astype(np.uint8))
        bbox = (x, y, w, h)

        return img, mask, {"type": DefectClass.STAIN.value, "bbox": bbox}

    def generate_sample(self, defect_type: DefectClass, texture: str = "metal") -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generates a sample image, ground-truth mask, and metadata record."""
        surface = self.generate_base_texture(texture)

        if defect_type == DefectClass.NORMAL:
            mask = np.zeros_like(surface, dtype=np.uint8)
            meta = {
                "type": DefectClass.NORMAL.value,
                "is_defective": False,
                "defects": []
            }
            return cv2.cvtColor(surface, cv2.COLOR_GRAY2BGR), mask, meta

        if defect_type == DefectClass.SCRATCH:
            img, mask, defect_info = self.inject_scratch(surface)
        elif defect_type == DefectClass.CRACK:
            img, mask, defect_info = self.inject_crack(surface)
        elif defect_type == DefectClass.PIT:
            img, mask, defect_info = self.inject_pit(surface)
        elif defect_type == DefectClass.STAIN:
            img, mask, defect_info = self.inject_stain(surface)
        else:
            img, mask, defect_info = self.inject_scratch(surface)

        meta = {
            "type": defect_type.value,
            "is_defective": True,
            "defects": [defect_info]
        }
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return bgr, mask, meta

    def generate_batch(self, count_per_class: int = 4, output_dir: Path = Path("data/samples")) -> List[Path]:
        """Generates a complete balanced dataset folder with images, masks, and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        created_paths: List[Path] = []

        classes = [
            DefectClass.NORMAL,
            DefectClass.SCRATCH,
            DefectClass.CRACK,
            DefectClass.PIT,
            DefectClass.STAIN
        ]
        textures = ["metal", "cast_iron", "ceramic"]

        idx = 1
        for d_class in classes:
            for i in range(count_per_class):
                tex = textures[(idx + i) % len(textures)]
                img, mask, meta = self.generate_sample(d_class, texture=tex)

                base_name = f"sample_{idx:03d}_{d_class.value.lower()}"
                img_path = output_dir / f"{base_name}.png"
                mask_path = output_dir / f"{base_name}_mask.png"
                meta_path = output_dir / f"{base_name}_meta.json"

                meta["image_file"] = img_path.name
                meta["mask_file"] = mask_path.name
                meta["texture"] = tex

                save_image(img_path, img)
                save_image(mask_path, mask)
                save_json(meta_path, meta)

                created_paths.append(img_path)
                idx += 1

        return created_paths
