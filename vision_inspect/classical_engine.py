"""
Classical Computer Vision Defect Detection Engine.
Extracts morphological top-hat/black-hat anomalies, contour geometry, and GLCM texture features.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import cv2
import numpy as np
from .config import ClassicalEngineConfig, DefectClass


@dataclass
class CandidateRegion:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: float
    perimeter: float
    aspect_ratio: float
    circularity: float
    solidity: float
    extent: float
    mean_intensity_diff: float
    predicted_class: DefectClass
    confidence: float
    contour: np.ndarray
    mask: np.ndarray


class ClassicalDefectEngine:
    """Detects surface anomalies using classical image processing, morphology, and contour geometry."""

    def __init__(self, config: ClassicalEngineConfig = None):
        self.config = config or ClassicalEngineConfig()

    def morphological_enhancement(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts morphological Top-Hat (bright anomalies) and Black-Hat (dark anomalies).
        """
        k_sz = max(15, self.config.blackhat_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sz, k_sz))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        return tophat, blackhat

    def segment_candidates(self, gray: np.ndarray) -> np.ndarray:
        """
        Combines morphological residual filters with multi-scale background subtraction
        to isolate both high-frequency defects (scratches, pits, cracks) and diffuse defects (stains).
        """
        # 1. High-frequency structural anomalies
        tophat, blackhat = self.morphological_enhancement(gray)
        combined_residue = cv2.add(tophat, blackhat)

        mean_res = float(np.mean(combined_residue))
        std_res = float(np.std(combined_residue))
        thresh_val = max(24.0, mean_res + 2.8 * std_res)
        _, binary_struct = cv2.threshold(combined_residue, thresh_val, 255, cv2.THRESH_BINARY)
        binary_struct = binary_struct.astype(np.uint8)

        # 2. Diffuse low-frequency anomalies (stains / smudges)
        med_bg = cv2.medianBlur(gray, 31)
        diffuse_residue = cv2.subtract(med_bg, gray)
        _, binary_diffuse = cv2.threshold(diffuse_residue, 18, 255, cv2.THRESH_BINARY)
        # Filter small noise from diffuse channel
        k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_diffuse = cv2.morphologyEx(binary_diffuse, cv2.MORPH_OPEN, k_clean)

        # Merge structural and diffuse detections
        combined_binary = cv2.bitwise_or(binary_struct, binary_diffuse)

        # Morphological opening and closing
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morph_open_kernel, self.config.morph_open_kernel))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morph_close_kernel, self.config.morph_close_kernel))

        opened = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, k_open)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
        return closed

    def compute_glcm_proxy_features(self, patch: np.ndarray) -> Dict[str, float]:
        """Calculates second-order statistical texture features on local patch."""
        if patch.size < 4:
            return {"contrast": 0.0, "homogeneity": 1.0, "energy": 0.0}

        quantized = (patch / 16).astype(np.uint8)
        h, w = quantized.shape
        glcm = np.zeros((16, 16), dtype=np.float32)

        for i in range(h):
            for j in range(w - 1):
                p1, p2 = quantized[i, j], quantized[i, j + 1]
                glcm[p1, p2] += 1.0

        total = np.sum(glcm)
        if total > 0:
            glcm /= total

        rows, cols = np.indices((16, 16))
        contrast = float(np.sum(glcm * ((rows - cols) ** 2)))
        homogeneity = float(np.sum(glcm / (1.0 + np.abs(rows - cols))))
        energy = float(np.sum(glcm ** 2))
        return {"contrast": contrast, "homogeneity": homogeneity, "energy": energy}

    def classify_contour(
        self,
        contour: np.ndarray,
        gray: np.ndarray,
        binary_mask: np.ndarray
    ) -> CandidateRegion:
        """Extracts geometric descriptors and assigns defect class based on shape invariants."""
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)

        # Shape descriptors
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        circularity = (4.0 * np.pi * area) / ((perimeter ** 2) + 1e-5)
        circularity = min(1.0, circularity)

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / (hull_area + 1e-5)
        extent = area / (float(w * h) + 1e-5)

        # Contrast
        region_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=-1)
        defect_mean = cv2.mean(gray, mask=region_mask)[0]

        dilated = cv2.dilate(region_mask, np.ones((7, 7), np.uint8))
        rim_mask = cv2.subtract(dilated, region_mask)
        rim_mean = cv2.mean(gray, mask=rim_mask)[0] if cv2.countNonZero(rim_mask) > 0 else defect_mean
        mean_diff = abs(defect_mean - rim_mean)

        patch = gray[y:y+h, x:x+w]
        texture = self.compute_glcm_proxy_features(patch)

        # Physical geometry heuristics
        if aspect_ratio > 3.0:
            if solidity < 0.65 or perimeter > 2.8 * (w + h):
                predicted = DefectClass.CRACK
                confidence = min(0.96, 0.60 + 0.35 * (1.0 - solidity))
            else:
                predicted = DefectClass.SCRATCH
                confidence = min(0.96, 0.55 + 0.40 * (aspect_ratio / 8.0))
        elif circularity > 0.52 and area < 800:
            predicted = DefectClass.PIT
            confidence = min(0.95, 0.60 + 0.35 * circularity)
        elif area > 120 and mean_diff < 40.0 and aspect_ratio < 2.5:
            predicted = DefectClass.STAIN
            confidence = min(0.92, 0.55 + 0.35 * (1.0 - abs(circularity - 0.5)))
        else:
            if aspect_ratio > 2.0:
                predicted = DefectClass.SCRATCH
                confidence = 0.55
            else:
                predicted = DefectClass.STAIN if area > 200 else DefectClass.PIT
                confidence = 0.50

        single_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(single_mask, [contour], -1, 255, thickness=-1)

        return CandidateRegion(
            bbox=(int(x), int(y), int(w), int(h)),
            area=area,
            perimeter=perimeter,
            aspect_ratio=float(aspect_ratio),
            circularity=float(circularity),
            solidity=float(solidity),
            extent=float(extent),
            mean_intensity_diff=float(mean_diff),
            predicted_class=predicted,
            confidence=float(confidence),
            contour=contour,
            mask=single_mask
        )

    def detect(self, gray: np.ndarray) -> Tuple[List[CandidateRegion], np.ndarray]:
        """Runs candidate detection and returns list of candidate regions and full binary defect mask."""
        binary_mask = self.segment_candidates(gray)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[CandidateRegion] = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.config.min_contour_area <= area <= self.config.max_contour_area:
                candidate = self.classify_contour(c, gray, binary_mask)
                candidates.append(candidate)

        return candidates, binary_mask
