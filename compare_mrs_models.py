"""Evaluate YOLOv5n, YOLOv8n, YOLOv12n, and MDI-YOLO on MRSDataset.

All models are evaluated with identical settings. Per-model validation artifacts and
comparison_results.csv/json are written to runs/model_comparison by default.

Example:
    python compare_mrs_models.py --device 0
    python compare_mrs_models.py --device cpu --batch 1 --no-plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

import yaml

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "MRSDataset.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "model_comparison"

MODELS = {
    "YOLOv8n": Path(r"E:\CODE\ultralytics-main\runs\detect\train11\weights\best.pt"),
    "YOLOv5n": Path(r"E:\CODE\ultralytics-main\runs\detect\train12\weights\best.pt"),
    "YOLOv12n": Path(r"E:\Desktop\model_compare\yolov12\runs\train\mrs_yolov12n2\weights\best.pt"),
    "MDI-YOLO": Path(r"E:\Desktop\model_compare\ultralytics-main\runs\train\mrs_yolov8\weights\best.pt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare four trained detectors on the same MRSDataset split.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="MRSDataset YAML path.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root override. By default, use the path value from --data.",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=960, help="Common inference image size.")
    parser.add_argument("--batch", type=int, default=4, help="Common validation batch size.")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 is safest on Windows).")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold used for metrics.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Comparison output directory.")
    parser.add_argument("--no-plots", action="store_true", help="Do not save validation plots.")
    return parser.parse_args()


def make_runtime_data_yaml(source: Path, dataset_root: Path | None, output: Path, split: str) -> Path:
    """Validate the data YAML and optionally create a copy with an overridden root."""
    if not source.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {source}")

    with source.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if split not in config:
        raise KeyError(f"Split '{split}' is absent from {source}")

    if dataset_root is not None:
        dataset_root = dataset_root.resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root override not found: {dataset_root}")
        config["path"] = str(dataset_root)
    else:
        configured_root = Path(config.get("path", source.parent))
        if not configured_root.is_absolute():
            configured_root = source.parent / configured_root
        if not configured_root.is_dir():
            raise FileNotFoundError(
                f"Dataset root from {source} not found: {configured_root}\n"
                "Fix the YAML 'path' value or pass --dataset-root with the actual directory."
            )

    output.mkdir(parents=True, exist_ok=True)
    runtime_yaml = output / "resolved_MRSDataset.yaml"
    with runtime_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return runtime_yaml


def number(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_result(name: str, weights: Path, model: YOLO, metrics, elapsed: float, imgsz: int) -> dict:
    values = getattr(metrics, "results_dict", {}) or {}
    precision = number(values.get("metrics/precision(B)", getattr(metrics.box, "mp", None)))
    recall = number(values.get("metrics/recall(B)", getattr(metrics.box, "mr", None)))
    speed = getattr(metrics, "speed", {}) or {}
    params = sum(parameter.numel() for parameter in model.model.parameters())
    flops = number(get_flops(model.model, imgsz))
    return {
        "model": name,
        "status": "ok",
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mAP50": number(values.get("metrics/mAP50(B)", metrics.box.map50)),
        "mAP50-95": number(values.get("metrics/mAP50-95(B)", metrics.box.map)),
        "parameters": params,
        "GFLOPs": flops,
        "preprocess_ms_per_image": number(speed.get("preprocess")),
        "inference_ms_per_image": number(speed.get("inference")),
        "postprocess_ms_per_image": number(speed.get("postprocess")),
        "total_elapsed_seconds": elapsed,
        "weights_mb": weights.stat().st_size / (1024**2),
        "weights": str(weights),
        "save_dir": str(metrics.save_dir),
        "error": "",
    }


def save_results(rows: list[dict], output: Path) -> None:
    csv_path = output / "comparison_results.csv"
    json_path = output / "comparison_results.json"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2, allow_nan=True)
    print(f"\nComparison CSV:  {csv_path}")
    print(f"Comparison JSON: {json_path}")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    dataset_root = args.dataset_root.resolve() if args.dataset_root is not None else None
    data = make_runtime_data_yaml(args.data.resolve(), dataset_root, output, args.split)

    missing = [f"{name}: {path}" for name, path in MODELS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model weights:\n" + "\n".join(missing))

    rows = []
    for name, weights in MODELS.items():
        print("\n" + "=" * 80)
        print(f"Evaluating {name}: {weights}")
        print("=" * 80)
        try:
            model = YOLO(str(weights))
            started = time.perf_counter()
            metrics = model.val(
                data=str(data),
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                project=str(output),
                name=name,
                exist_ok=True,
                plots=not args.no_plots,
                verbose=True,
            )
            rows.append(extract_result(name, weights, model, metrics, time.perf_counter() - started, args.imgsz))
        except Exception as exc:  # Continue so one incompatible checkpoint does not discard other results.
            traceback.print_exc()
            rows.append(
                {
                    "model": name,
                    "status": "failed",
                    "precision": math.nan,
                    "recall": math.nan,
                    "f1": math.nan,
                    "mAP50": math.nan,
                    "mAP50-95": math.nan,
                    "parameters": math.nan,
                    "GFLOPs": math.nan,
                    "preprocess_ms_per_image": math.nan,
                    "inference_ms_per_image": math.nan,
                    "postprocess_ms_per_image": math.nan,
                    "total_elapsed_seconds": math.nan,
                    "weights_mb": weights.stat().st_size / (1024**2),
                    "weights": str(weights),
                    "save_dir": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        save_results(rows, output)

    print("\nFinal results")
    for row in rows:
        if row["status"] == "ok":
            print(
                f"{row['model']:<10} P={row['precision']:.4f} R={row['recall']:.4f} "
                f"mAP50={row['mAP50']:.4f} mAP50-95={row['mAP50-95']:.4f} "
                f"inference={row['inference_ms_per_image']:.2f} ms/image"
            )
        else:
            print(f"{row['model']:<10} FAILED: {row['error']}")


if __name__ == "__main__":
    main()
