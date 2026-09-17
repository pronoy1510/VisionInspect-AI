"""
Evaluation and Benchmarking Module for VisionInspect-AI.
Computes classification metrics, segmentation mIoU, Dice scores, and system latency profile.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .config import DefectClass, InspectionConfig
from .detector import HybridDefectDetector, InspectionResult
from .utils import load_image, load_json, save_image, save_json, setup_logger
from .visualizer import InspectionVisualizer


class ModelEvaluator:
    """Evaluates detector against labeled dataset directory."""

    def __init__(self, detector: Optional[HybridDefectDetector] = None):
        self.detector = detector or HybridDefectDetector()
        self.logger = setup_logger("ModelEvaluator")

    def compute_iou(self, mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
        """Computes Intersection over Union between two binary masks."""
        pred_bool = mask_pred > 0
        gt_bool = mask_gt > 0
        intersection = np.logical_and(pred_bool, gt_bool).sum()
        union = np.logical_or(pred_bool, gt_bool).sum()
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return float(intersection / union)

    def evaluate_directory(self, data_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Runs evaluation on all samples in directory with matching metadata and ground-truth masks.
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir or Path("data/results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        meta_files = sorted(list(data_dir.glob("*_meta.json")))
        if not meta_files:
            raise FileNotFoundError(f"No metadata files (*_meta.json) found in {data_dir.resolve()}")

        y_true: List[str] = []
        y_pred: List[str] = []
        ious: List[float] = []
        latencies: List[float] = []
        records: List[Dict[str, Any]] = []

        class_labels = DefectClass.list_classes()

        for meta_p in meta_files:
            meta = load_json(meta_p)
            base_stem = meta_p.stem.replace("_meta", "")
            img_path = data_dir / f"{base_stem}.png"
            mask_path = data_dir / f"{base_stem}_mask.png"

            if not img_path.exists():
                continue

            true_label = meta.get("type", DefectClass.NORMAL.value)
            y_true.append(true_label)

            # Inference
            result: InspectionResult = self.detector.inspect(img_path)
            y_pred.append(result.primary_defect)
            latencies.append(result.latency_ms)

            # IoU if mask exists
            iou = 0.0
            if mask_path.exists():
                gt_mask = load_image(mask_path, grayscale=True)
                if result.binary_mask is not None:
                    # resize mask to match gt if needed
                    pred_m = result.binary_mask
                    if pred_m.shape != gt_mask.shape:
                        pred_m = cv2.resize(pred_m, (gt_mask.shape[1], gt_mask.shape[0]))
                    iou = self.compute_iou(pred_m, gt_mask)
                    ious.append(iou)

            records.append({
                "sample": base_stem,
                "ground_truth": true_label,
                "prediction": result.primary_defect,
                "is_defective_gt": meta.get("is_defective", False),
                "is_defective_pred": result.is_defective,
                "severity": result.overall_severity,
                "defect_count": result.defect_count,
                "iou": round(iou, 4),
                "latency_ms": round(result.latency_ms, 2)
            })

            # Save visual inspection card
            bgr = load_image(img_path)
            panel = InspectionVisualizer.create_inspection_panel(bgr, result)
            InspectionVisualizer.draw_detection_overlay(bgr, result)
            save_image(output_dir / f"{base_stem}_panel.png", panel)

        # Classification metrics
        acc = float(accuracy_score(y_true, y_pred))
        p, r, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=class_labels, zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        InspectionVisualizer.plot_confusion_matrix(cm, class_labels, output_dir / "confusion_matrix.png")

        per_class_metrics = {}
        for idx, cl in enumerate(class_labels):
            per_class_metrics[cl] = {
                "precision": round(float(p[idx]), 4),
                "recall": round(float(r[idx]), 4),
                "f1": round(float(f1[idx]), 4),
                "support": int(support[idx])
            }

        lat_arr = np.array(latencies)
        summary = {
            "total_samples": len(y_true),
            "accuracy": round(acc, 4),
            "macro_f1": round(float(np.mean(f1)), 4),
            "mean_iou": round(float(np.mean(ious)) if ious else 0.0, 4),
            "latency": {
                "mean_ms": round(float(np.mean(lat_arr)), 2),
                "median_ms": round(float(np.median(lat_arr)), 2),
                "p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
                "fps": round(1000.0 / float(np.mean(lat_arr)), 1)
            },
            "per_class": per_class_metrics,
            "confusion_matrix": cm.tolist()
        }

        # Save summary JSON and CSV
        save_json(output_dir / "evaluation_summary.json", summary)
        df = pd.DataFrame(records)
        df.to_csv(output_dir / "evaluation_details.csv", index=False)

        return summary
