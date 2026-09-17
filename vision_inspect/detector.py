"""
Unified Hybrid Defect Detection and Quality Assessment Engine.
Fuses classical contour geometry with deep reconstruction anomaly heatmaps.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from .config import DefectClass, InspectionConfig
from .preprocessing import ImagePreprocessor
from .classical_engine import ClassicalDefectEngine, CandidateRegion
from .deep_engine import DeepAnomalyEngine
from .utils import load_image, setup_logger


@dataclass
class DefectRegion:
    defect_type: DefectClass
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: float
    severity_label: str
    mask: np.ndarray


@dataclass
class InspectionResult:
    image_path: Optional[str]
    is_defective: bool
    primary_defect: str
    defect_count: int
    overall_severity: str
    global_anomaly_score: float
    defects: List[DefectRegion] = field(default_factory=list)
    latency_ms: float = 0.0
    preprocessed_img: Optional[np.ndarray] = None
    anomaly_heatmap: Optional[np.ndarray] = None
    binary_mask: Optional[np.ndarray] = None

    def to_dict(self):
        return {
            "image_path": str(self.image_path) if self.image_path else "memory",
            "is_defective": self.is_defective,
            "primary_defect": self.primary_defect,
            "defect_count": self.defect_count,
            "overall_severity": self.overall_severity,
            "global_anomaly_score": round(self.global_anomaly_score, 4),
            "latency_ms": round(self.latency_ms, 2),
            "defects": [
                {
                    "type": d.defect_type.value,
                    "confidence": round(d.confidence, 4),
                    "bbox": list(d.bbox),
                    "area": round(d.area, 2),
                    "severity": d.severity_label
                }
                for d in self.defects
            ]
        }


class HybridDefectDetector:
    """Production-grade hybrid inspector combining classical and deep vision techniques."""

    def __init__(self, config: Optional[InspectionConfig] = None):
        self.config = config or InspectionConfig()
        self.logger = setup_logger("HybridDefectDetector")
        self.preprocessor = ImagePreprocessor(self.config.preprocess)
        self.classical_engine = ClassicalDefectEngine(self.config.classical)
        self.deep_engine = DeepAnomalyEngine(self.config.deep)

    def _nms_bboxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
        iou_threshold: float = 0.30
    ) -> List[int]:
        """Performs Non-Maximum Suppression on bounding boxes."""
        if not boxes:
            return []

        boxes_arr = np.array(boxes, dtype=np.float32)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = x1 + boxes_arr[:, 2]
        y2 = y1 + boxes_arr[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def inspect(self, image_input: Union[str, Path, np.ndarray]) -> InspectionResult:
        """
        Executes end-to-end surface quality inspection on an image path or numpy array.
        """
        start_time = time.perf_counter()
        img_path_str = None

        if isinstance(image_input, (str, Path)):
            img_path_str = str(image_input)
            img = load_image(image_input)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise TypeError("image_input must be a file path or numpy ndarray")

        # Step 1: Preprocessing
        enhanced_gray, norm_float, _ = self.preprocessor.preprocess(img)

        # Step 2: Classical Defect Candidate Extraction
        classical_candidates, binary_mask = self.classical_engine.detect(enhanced_gray)

        # Step 3: Deep Anomaly Residual Heatmap
        heatmap, global_anomaly_score = self.deep_engine.compute_anomaly_map(norm_float)

        # Step 4: Hybrid Fusion and Verification
        raw_boxes: List[Tuple[int, int, int, int]] = []
        raw_scores: List[float] = []
        raw_types: List[DefectClass] = []
        raw_masks: List[np.ndarray] = []
        raw_areas: List[float] = []

        h_img, w_img = enhanced_gray.shape[:2]

        for cand in classical_candidates:
            x, y, w, h = cand.bbox
            patch_heat = heatmap[y:y+h, x:x+w]
            deep_intensity = float(np.mean(patch_heat)) if patch_heat.size > 0 else 0.0

            # Fused confidence weighting
            fused_conf = (
                self.config.fusion_classical_weight * cand.confidence +
                self.config.fusion_deep_weight * max(deep_intensity, cand.confidence * 0.8)
            )

            # Intensity deviation filter: allow structural defects with contrast or larger diffuse stains
            if (cand.mean_intensity_diff >= 8.0 or cand.area > 200) and fused_conf >= self.config.min_confidence:
                raw_boxes.append((x, y, w, h))
                raw_scores.append(fused_conf)
                raw_types.append(cand.predicted_class)
                raw_masks.append(cand.mask)
                raw_areas.append(cand.area)

        # Step 5: Non-Maximum Suppression
        keep_indices = self._nms_bboxes(raw_boxes, raw_scores, iou_threshold=0.30)

        final_defects: List[DefectRegion] = []
        combined_defect_mask = np.zeros_like(enhanced_gray, dtype=np.uint8)

        for idx in keep_indices:
            bbox = raw_boxes[idx]
            conf = raw_scores[idx]
            dtype = raw_types[idx]
            mask = raw_masks[idx]
            area = raw_areas[idx]

            combined_defect_mask = cv2.bitwise_or(combined_defect_mask, mask)

            if conf >= self.config.severity_high_threshold or area > 1000:
                sev = "Critical"
            elif conf >= self.config.severity_medium_threshold or area > 300:
                sev = "Severe"
            else:
                sev = "Moderate"

            final_defects.append(
                DefectRegion(
                    defect_type=dtype,
                    confidence=conf,
                    bbox=bbox,
                    area=area,
                    severity_label=sev,
                    mask=mask
                )
            )

        # Determine overall surface classification & severity
        total_defect_area = sum(d.area for d in final_defects)
        area_fraction = total_defect_area / (float(h_img * w_img) + 1e-6)
        is_defective = len(final_defects) > 0

        if not is_defective:
            overall_severity = "Pristine"
            primary_defect = DefectClass.NORMAL.value
        else:
            # Pick highest confidence / largest defect as primary
            best_defect = max(final_defects, key=lambda d: (d.confidence, d.area))
            primary_defect = best_defect.defect_type.value

            if area_fraction > 0.05 or any(d.severity_label == "Critical" for d in final_defects):
                overall_severity = "Critical"
            elif area_fraction > 0.015 or any(d.severity_label == "Severe" for d in final_defects):
                overall_severity = "Severe"
            else:
                overall_severity = "Moderate"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return InspectionResult(
            image_path=img_path_str,
            is_defective=is_defective,
            primary_defect=primary_defect,
            defect_count=len(final_defects),
            overall_severity=overall_severity,
            global_anomaly_score=global_anomaly_score,
            defects=final_defects,
            latency_ms=elapsed_ms,
            preprocessed_img=enhanced_gray,
            anomaly_heatmap=heatmap,
            binary_mask=combined_defect_mask
        )
