# VisionInspect-AI: Industrial Surface Defect Detection & Quality Assessment Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![OpenCV 4.8+](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Test Suite](https://img.shields.io/badge/pytest-10%20passed-brightgreen.svg)]()
[![Throughput](https://img.shields.io/badge/latency-12.2ms%20%7C%2082.2%20FPS-purple.svg)]()

> **VisionInspect-AI** is an industrial-grade Computer Vision pipeline engineered for autonomous surface defect detection, anomaly localization, and quality assurance triage. It combines classical morphological feature extraction (Top-Hat/Black-Hat, adaptive thresholding, GLCM texture features) with a deep convolutional autoencoder anomaly reconstruction engine.

---

## 1. Project Overview

Modern high-precision manufacturing lines (aerospace composites, semiconductor wafers, metallurgy, ceramic production) demand real-time optical inspection to intercept structural flaws before shipping. 

**VisionInspect-AI** addresses the vulnerability of single-paradigm systems:
- Purely classical thresholding algorithms fail under uneven factory lighting, background grain, and complex surface finishes.
- Heavyweight end-to-end deep networks demand expensive GPUs, require thousands of labeled defect training samples, and suffer from high latency.

**Our Solution**: A **hybrid vision framework** that pre-processes surfaces using CLAHE and bilateral filtering, extracts candidate anomaly contours using multi-scale morphological operators, evaluates structural reconstruction residuals via an unsupervised Convolutional Autoencoder, and fuses spatial descriptors with deep heatmap intensities via Non-Maximum Suppression (NMS).

---

## 2. Key Features

- **Dual-Engine Hybrid Fusion**: Fuses geometric shape invariants (aspect ratio, circularity, solidity) with deep reconstruction loss (L1 residual maps).
- **Multi-Class Defect Taxonomy**: Detects and classifies:
  - `Scratch`: High aspect ratio, specular highlight, linear/curved channel.
  - `Crack`: Jagged random-walk fracture path, low solidity, high perimeter roughness.
  - `Pit / Void`: High circularity, localized crater depth with rim shadows.
  - `Stain / Smudge`: Diffuse low-contrast organic or oil contamination.
  - `Normal`: Pristine surface with zero anomalies.
- **Explainable 4-Panel Diagnostics**: Generates multi-panel diagnostic boards showing:
  1. Raw Input Surface
  2. CLAHE Illumination-Enhanced Image
  3. PyTorch Deep Anomaly Heatmap (COLORMAP_JET)
  4. Final Bounding Box & Contour Overlay with Pass/Reject Status Banner
- **Real-Time Edge Throughput**: Profiles at **~12.2 ms per frame (>80 FPS)** on standard CPU hardware.
- **Parametric Synthetic Dataset Generator**: Injects realistic calibrated defects onto brushed metal, ceramic, and cast iron textures with pixel-level ground truth masks.
- **100% Terminal-Executable**: Full CLI interface for synthesis, single/batch inspection, quantitative evaluation, and benchmarking.
- **Comprehensive Verification Suite**: Complete pytest test coverage spanning all 5 sub-packages.

---

## 3. Technologies & Tools Used

| Domain | Technology / Library | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | >= 3.10 | Core programming runtime |
| **Computer Vision** | OpenCV (`opencv-python`) | >= 4.8.0 | Image filtering, CLAHE, morphology, contour analysis |
| **Deep Learning** | PyTorch (`torch`) | >= 2.0.0 | Lightweight Convolutional Autoencoder for anomaly heatmaps |
| **Scientific Computing** | NumPy & SciPy | >= 1.24.0 | Numerical array manipulation and spatial statistics |
| **Machine Learning** | Scikit-learn | >= 1.3.0 | Metrics evaluation (F1, Precision, Recall, Confusion Matrix) |
| **Data & Tabulation** | Pandas | >= 2.0.0 | Structured CSV inspection log serialization |
| **Visualization** | Matplotlib | >= 3.7.0 | Heatmap rendering and confusion matrix plots |
| **Testing** | Pytest | >= 7.4.0 | Automated unit and integration testing |
| **Documentation** | ReportLab | >= 4.0.0 | Automated generation of submission PDF report |

---

## 4. System Architecture

```
                                  [ Input Surface Image ]
                                             ?
                                             ?
                               [ Step 1: Preprocessing ]
                           - Aspect Ratio Preserving Resize
                           - Grayscale Conversion
                           - CLAHE Illumination Balance
                           - Bilateral Edge-Preserving Denoising
                                             ?
                       ?????????????????????????????????????????????
                       ?                                           ?
         [ Classical Defect Engine ]                   [ Deep Anomaly Engine ]
   - Multi-scale Top-Hat / Black-Hat             - Conv Autoencoder Inference
   - Statistical Residual Thresholding           - Pixel-wise L1 Reconstruction Error
   - Morphological Open / Close Cleanup          - Gaussian Residual Smoothing
   - Contour Geometry Extraction                 - Anomaly Heatmap Normalization
     (Aspect Ratio, Circularity, Solidity)                         ?
   - GLCM Texture Feature Profiling                                ?
                       ?                                           ?
                       ?????????????????????????????????????????????
                                             ?
                                             ?
                               [ Step 4: Hybrid Fusion ]
                         - Defect Candidate Spatial Matching
                         - Fused Confidence Weighting
                         - Non-Maximum Suppression (NMS)
                         - Severity Matrix Triage (Pristine / Minor / Severe / Critical)
                                             ?
                                             ?
                                  [ Outputs & Artifacts ]
                         - Annotated Visual Overlays (PNG)
                         - 4-Panel Diagnostic Dashboards (PNG)
                         - Structured Inspection Logs (JSON / CSV)
                         - Evaluation Confusion Matrix (PNG)
```

---

## 5. Directory Structure

```
computer vision/
??? vision_inspect/                 # Core package
?   ??? __init__.py                 # Package initialization & exports
?   ??? config.py                   # Hyperparameters, paths & palettes
?   ??? utils.py                    # I/O, JSON serialization, logging
?   ??? preprocessing.py            # CLAHE, bilateral smoothing, gradients
?   ??? classical_engine.py         # Morphology, contours, GLCM features
?   ??? deep_engine.py              # PyTorch Convolutional Autoencoder
?   ??? detector.py                 # Hybrid fusion detector & NMS
?   ??? dataset_generator.py        # Procedural texture & defect synthesis
?   ??? visualizer.py               # 4-panel dashboards & confusion matrices
?   ??? evaluator.py                # Precision, Recall, F1, mIoU, latency
??? tests/                          # Automated Pytest test suite
?   ??? test_preprocessing.py       # Preprocessor unit tests
?   ??? test_classical_engine.py    # Morphological defect extraction tests
?   ??? test_deep_engine.py         # PyTorch autoencoder tensor tests
?   ??? test_detector.py            # Hybrid detector integration tests
?   ??? test_dataset_generator.py   # Dataset synthesis verification tests
??? data/
?   ??? samples/                    # Generated sample images & ground truth
?   ??? results/                    # Output visual cards, metrics & CSVs
??? docs/                           # Documentation & architecture assets
??? main.py                         # Unified CLI entrypoint
??? generate_report.py              # Automated 15-section PDF report generator
??? statement.md                    # Project statement and scope
??? PROJECT_REPORT.md               # Full Markdown submission report
??? Project_Report.pdf              # Compiled submission PDF report
??? requirements.txt                # Python package dependencies
??? README.md                       # Main project documentation
```

---

## 6. Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/VisionInspect-AI.git
cd VisionInspect-AI
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. Execution & CLI Commands

VisionInspect-AI is fully executable via the command line with four primary subcommands:

### 1. Synthesize Benchmark Dataset
Generates synthetic surface samples across brushed metal, cast iron, and ceramic textures with calibrated ground-truth masks:
```bash
python main.py generate --count-per-class 4 --output data/samples
```

### 2. Inspect Single Image
Inspects an image, outputs defect classification, bounding boxes, severity rating, and saves the 4-panel diagnostic dashboard:
```bash
python main.py inspect --input data/samples/sample_005_scratch.png --output data/results
```
**Sample Output:**
```
[INFO] [CLI-Inspect] File: sample_005_scratch.png
[INFO] [CLI-Inspect] Status: DEFECTIVE
[INFO] [CLI-Inspect] Primary Defect: Scratch | Severity: Critical
[INFO] [CLI-Inspect] Defect Count: 1 | Anomaly Score: 0.1420
[INFO] [CLI-Inspect] Processing Time: 13.85 ms
[INFO] [CLI-Inspect] Inspection artifacts saved to data/results
```

### 3. Batch Inspect Image Directory
Runs automated batch inspection over all images in a target directory and writes a consolidated summary JSON:
```bash
python main.py inspect --dir data/samples --output data/results
```

### 4. Evaluate Dataset & Benchmark Accuracy
Computes full classification metrics (Accuracy, Precision, Recall, F1), mask localization (mIoU), and generates the confusion matrix plot:
```bash
python main.py evaluate --data-dir data/samples --output data/results
```

### 5. Profile Latency & Throughput
Measures min, max, median, 95th percentile latency, and Frames Per Second (FPS):
```bash
python main.py benchmark --iterations 25
```
**Sample Benchmark Log:**
```
==================================================
        VISIONINSPECT-AI BENCHMARK REPORT         
==================================================
Iterations      : 25
Mean Latency    : 12.17 ms
Median Latency  : 12.28 ms
Min Latency     : 10.35 ms
Max Latency     : 14.42 ms
95th Percentile : 13.96 ms
Throughput      : 82.2 Frames Per Second (FPS)
==================================================
```

---

## 8. Running the Automated Test Suite

Execute the entire test suite using `pytest`:
```bash
python -m pytest -v
```

**Expected Test Output:**
```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\...\computer vision
collected 10 items

tests/test_classical_engine.py::test_classical_engine_scratch_detection PASSED [ 10%]
tests/test_classical_engine.py::test_classical_engine_pit_detection PASSED     [ 20%]
tests/test_dataset_generator.py::test_surface_generator_types PASSED         [ 30%]
tests/test_dataset_generator.py::test_batch_generator PASSED                 [ 40%]
tests/test_deep_engine.py::test_autoencoder_tensor_shapes PASSED             [ 50%]
tests/test_deep_engine.py::test_deep_anomaly_engine_inference PASSED         [ 60%]
tests/test_detector.py::test_detector_normal_surface PASSED                  [ 70%]
tests/test_detector.py::test_detector_defective_surface PASSED               [ 80%]
tests/test_preprocessing.py::test_to_grayscale_and_resize PASSED             [ 90%]
tests/test_preprocessing.py::test_preprocess_pipeline PASSED                 [100%]

============================= 10 passed in 3.72s ==============================
```

---

## 9. Re-Compiling the PDF Project Report

The complete, 15-section project report can be recompiled into `Project_Report.pdf` at any time using:
```bash
python scripts/generate_report.py
```

---

## 10. Submission Guidelines Checklist

- [x] **Public Visibility**: The GitHub repository must be set to Public.
- [x] **Repository URL Format**: `https://github.com/{github-username}/{repo-name}` (strict format, no `/tree/` or `/blob/`).
- [x] **Root README.md**: Clear step-by-step setup, installation, CLI execution, and test commands.
- [x] **Root statement.md**: Comprehensive problem statement, scope, target users, and features.
- [x] **Terminal-Executable**: 100% executable from command-line without GUI dependencies.
- [x] **PDF Project Report**: Generated and submitted as `Project_Report.pdf` covering all 15 required sections.
