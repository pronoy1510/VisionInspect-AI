"""
VisionInspect-AI CLI Entry Point.
Provides command-line commands for dataset synthesis, image inspection, evaluation, and latency benchmarking.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

from vision_inspect.config import InspectionConfig, DefectClass
from vision_inspect.detector import HybridDefectDetector
from vision_inspect.dataset_generator import SyntheticSurfaceGenerator
from vision_inspect.evaluator import ModelEvaluator
from vision_inspect.visualizer import InspectionVisualizer
from vision_inspect.utils import load_image, save_image, save_json, setup_logger


def cmd_generate(args):
    logger = setup_logger("CLI-Generate")
    logger.info(f"Generating synthetic dataset in {args.output}...")
    generator = SyntheticSurfaceGenerator(seed=args.seed)
    created = generator.generate_batch(count_per_class=args.count_per_class, output_dir=Path(args.output))
    logger.info(f"Generated {len(created)} sample surface images with ground truth masks and metadata.")


def cmd_inspect(args):
    logger = setup_logger("CLI-Inspect")
    detector = HybridDefectDetector()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        img_path = Path(args.input)
        if not img_path.exists():
            logger.error(f"Input file not found: {img_path}")
            sys.exit(1)

        result = detector.inspect(img_path)
        logger.info(f"File: {img_path.name}")
        logger.info(f"Status: {'DEFECTIVE' if result.is_defective else 'PASSED'}")
        logger.info(f"Primary Defect: {result.primary_defect} | Severity: {result.overall_severity}")
        logger.info(f"Defect Count: {result.defect_count} | Anomaly Score: {result.global_anomaly_score:.4f}")
        logger.info(f"Processing Time: {result.latency_ms:.2f} ms")

        # Save annotated image and 4-panel dashboard
        bgr = load_image(img_path)
        annotated = InspectionVisualizer.draw_detection_overlay(bgr, result)
        panel = InspectionVisualizer.create_inspection_panel(bgr, result)

        base_stem = img_path.stem
        save_image(out_dir / f"{base_stem}_annotated.png", annotated)
        save_image(out_dir / f"{base_stem}_panel.png", panel)
        save_json(out_dir / f"{base_stem}_result.json", result.to_dict())
        logger.info(f"Inspection artifacts saved to {out_dir.resolve()}")

    elif args.dir:
        input_dir = Path(args.dir)
        extensions = [".png", ".jpg", ".jpeg", ".bmp"]
        img_files = [p for p in input_dir.iterdir() if p.suffix.lower() in extensions and not p.name.endswith("_mask.png") and not p.name.endswith("_panel.png") and not p.name.endswith("_annotated.png")]

        logger.info(f"Batch inspecting {len(img_files)} images from {input_dir.resolve()}...")
        summary_results = []
        for p in img_files:
            res = detector.inspect(p)
            summary_results.append(res.to_dict())
            bgr = load_image(p)
            annotated = InspectionVisualizer.draw_detection_overlay(bgr, res)
            save_image(out_dir / f"{p.stem}_annotated.png", annotated)

        save_json(out_dir / "batch_inspection_summary.json", {"total_inspected": len(img_files), "results": summary_results})
        logger.info(f"Batch inspection complete. Summary saved to {out_dir / 'batch_inspection_summary.json'}")
    else:
        logger.error("Must provide either --input <filepath> or --dir <dirpath>")
        sys.exit(1)


def cmd_evaluate(args):
    logger = setup_logger("CLI-Evaluate")
    logger.info(f"Evaluating detector on dataset at {args.data_dir}...")
    evaluator = ModelEvaluator()
    summary = evaluator.evaluate_directory(Path(args.data_dir), Path(args.output))

    print("\n" + "="*50)
    print("      VISIONINSPECT-AI EVALUATION RESULTS        ")
    print("="*50)
    print(f"Total Evaluated Samples : {summary['total_samples']}")
    print(f"Overall Accuracy        : {summary['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score          : {summary['macro_f1']:.4f}")
    print(f"Mean IoU (Defect Masks) : {summary['mean_iou']:.4f}")
    print(f"Average Inference Latency: {summary['latency']['mean_ms']:.2f} ms ({summary['latency']['fps']} FPS)")
    print(f"P95 Inference Latency   : {summary['latency']['p95_ms']:.2f} ms")
    print("-"*50)
    print("Per-Class Metrics:")
    for cls_name, m in summary["per_class"].items():
        print(f"  - {cls_name:<10}: Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1={m['f1']:.2f} (n={m['support']})")
    print("="*50 + "\n")
    logger.info(f"Evaluation report and confusion matrix saved to {Path(args.output).resolve()}")


def cmd_benchmark(args):
    logger = setup_logger("CLI-Benchmark")
    detector = HybridDefectDetector()
    generator = SyntheticSurfaceGenerator(seed=123)
    sample, _, _ = generator.generate_sample(DefectClass.SCRATCH, texture="metal")

    # Warmup
    for _ in range(5):
        detector.inspect(sample)

    latencies = []
    logger.info(f"Running latency benchmark for {args.iterations} iterations...")
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        detector.inspect(sample)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(latencies)
    fps = 1000.0 / np.mean(lat_arr)
    print("\n" + "="*50)
    print("        VISIONINSPECT-AI BENCHMARK REPORT         ")
    print("="*50)
    print(f"Iterations      : {args.iterations}")
    print(f"Mean Latency    : {np.mean(lat_arr):.2f} ms")
    print(f"Median Latency  : {np.median(lat_arr):.2f} ms")
    print(f"Min Latency     : {np.min(lat_arr):.2f} ms")
    print(f"Max Latency     : {np.max(lat_arr):.2f} ms")
    print(f"95th Percentile : {np.percentile(lat_arr, 95):.2f} ms")
    print(f"Throughput      : {fps:.1f} Frames Per Second (FPS)")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        prog="visioninspect",
        description="VisionInspect-AI: Industrial Surface Defect Detection & Quality Assessment Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic defect dataset")
    gen_parser.add_argument("--count-per-class", type=int, default=4, help="Number of samples per class")
    gen_parser.add_argument("--output", type=str, default="data/samples", help="Output directory")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # inspect
    ins_parser = subparsers.add_parser("inspect", help="Run defect inspection on image or directory")
    ins_parser.add_argument("--input", type=str, help="Path to single image")
    ins_parser.add_argument("--dir", type=str, help="Path to directory of images")
    ins_parser.add_argument("--output", type=str, default="data/results", help="Directory to save visual overlays")

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate accuracy, IoU, and F1 on dataset")
    eval_parser.add_argument("--data-dir", type=str, default="data/samples", help="Directory containing samples and metadata")
    eval_parser.add_argument("--output", type=str, default="data/results", help="Directory to save evaluation artifacts")

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Profile inference latency and FPS throughput")
    bench_parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
