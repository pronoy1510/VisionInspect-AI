"""
Visualization and Diagnostic Overlay Module.
Renders bounding boxes, severity banners, contour masks, and multi-panel inspection boards.
"""

from pathlib import Path
from typing import Optional, Union
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import DEFECT_COLORS_BGR, DefectClass
from .detector import InspectionResult
from .utils import save_image


class InspectionVisualizer:
    """Builds visual diagnostic overlays and evaluation plots."""

    @staticmethod
    def draw_detection_overlay(img_bgr: np.ndarray, result: InspectionResult) -> np.ndarray:
        """
        Draws bounding boxes, defect labels, confidence badges, and top status banner.
        """
        annotated = img_bgr.copy()
        h, w = annotated.shape[:2]

        # Draw top status banner
        banner_h = 36
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        if result.is_defective:
            banner[:] = (35, 35, 180)  # Red banner
            status_text = f"REJECTED: {result.primary_defect.upper()} ({result.overall_severity}) | {result.latency_ms:.1f}ms"
        else:
            banner[:] = (35, 150, 45)  # Green banner
            status_text = f"PASSED: NORMAL (Pristine Surface) | {result.latency_ms:.1f}ms"

        cv2.putText(banner, status_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        annotated = np.vstack([banner, annotated])

        # Draw bounding boxes and defect badges (offset y by banner_h)
        for defect in result.defects:
            bx, by, bw, bh = defect.bbox
            by_offset = by + banner_h

            color = DEFECT_COLORS_BGR.get(defect.defect_type.value, (0, 255, 255))

            # Bounding box
            cv2.rectangle(annotated, (bx, by_offset), (bx + bw, by_offset + bh), color, 2)

            # Label badge
            label = f"{defect.defect_type.value} {int(defect.confidence * 100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                annotated,
                (bx, max(banner_h, by_offset - th - 6)),
                (bx + tw + 6, max(banner_h + th + 6, by_offset)),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (bx + 3, max(banner_h + 12, by_offset - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return annotated

    @staticmethod
    def create_inspection_panel(img_bgr: np.ndarray, result: InspectionResult) -> np.ndarray:
        """
        Creates a 4-panel diagnostic dashboard:
        [ Original Image | Enhanced Preprocess | Anomaly Heatmap | Detection Overlay ]
        """
        h, w = result.preprocessed_img.shape[:2] if result.preprocessed_img is not None else img_bgr.shape[:2]
        orig_resized = cv2.resize(img_bgr, (w, h))

        # Panel 1: Original
        p1 = orig_resized.copy()
        cv2.putText(p1, "1. Input Surface", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Panel 2: Preprocessed CLAHE
        if result.preprocessed_img is not None:
            p2 = cv2.cvtColor(result.preprocessed_img, cv2.COLOR_GRAY2BGR)
        else:
            p2 = cv2.cvtColor(cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        cv2.putText(p2, "2. CLAHE Filtered", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Panel 3: Anomaly Heatmap
        if result.anomaly_heatmap is not None:
            heat_uint8 = (result.anomaly_heatmap * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
            p3 = cv2.addWeighted(p2, 0.45, heat_color, 0.55, 0)
        else:
            p3 = p2.copy()
        cv2.putText(p3, f"3. Anomaly Heatmap (Err: {result.global_anomaly_score:.3f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Panel 4: Overlay
        annotated_full = InspectionVisualizer.draw_detection_overlay(orig_resized, result)
        p4 = cv2.resize(annotated_full, (w, h))

        # Stitch 2x2 grid
        top_row = np.hstack([p1, p2])
        bot_row = np.hstack([p3, p4])
        dashboard = np.vstack([top_row, bot_row])
        return dashboard

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path) -> None:
        """Plots and saves normalized confusion matrix as high-resolution PNG."""
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    x=j, y=i, s=f"{cm[i, j]}",
                    va="center", ha="center", size=11, weight="bold",
                    color="white" if cm[i, j] > np.max(cm) / 2 else "black"
                )

        fig.colorbar(cax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="left")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted Defect Category", weight="bold", labelpad=8)
        ax.set_ylabel("True Defect Category", weight="bold", labelpad=8)
        ax.set_title("VisionInspect-AI Confusion Matrix", weight="bold", pad=15)
        plt.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path))
        plt.close(fig)
