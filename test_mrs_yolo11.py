"""Evaluate the trained MRS YOLO detector on the test split.

Default checkpoint:
    runs/detect/yolo11n_SOSE_MAFF_precision_seed0/weights/best.pt

Example:
    python test_mrs_yolo11.py
    python test_mrs_yolo11.py --weights runs/detect/yolo11n_MAFF_seed0/weights/best.pt --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from typing_extensions import Self

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "yolo11n_SOSE_MAFF_precision_seed0" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "MRSDataset.yaml"


class SerialPool:
    """Small ThreadPool replacement for restricted Windows environments."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        return None

    @staticmethod
    def imap(func, iterable: Iterable):
        return map(func, iterable)


def force_serial_cache_scan() -> None:
    """Disable Ultralytics label-cache ThreadPool creation for this process."""
    import ultralytics.data.base as data_base
    import ultralytics.data.dataset as data_dataset

    data_base.ThreadPool = SerialPool
    data_dataset.ThreadPool = SerialPool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained YOLO detector on MRSDataset test split.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to trained .pt weights.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML containing the test split.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=4, help="Validation batch size.")
    parser.add_argument("--device", default="0", help="CUDA device id, 'cpu', or comma-separated device ids.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers. Use 0 on Windows for stability.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for metric calculation.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--name", default="mrs_test_yolo11n_SOSE_MAFF_precision_seed0", help="Output run name.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "detect", help="Output project directory.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-style JSON metrics when supported.")
    parser.add_argument("--no-plots", action="store_true", help="Disable PR/F1/confusion-matrix plots.")
    parser.add_argument(
        "--parallel-cache",
        action="store_true",
        help="Use Ultralytics' default parallel label-cache scanner instead of the safer serial scanner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    data = args.data.resolve()

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    if not args.parallel_cache:
        force_serial_cache_scan()

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=str(args.project),
        name=args.name,
        save_json=args.save_json,
        plots=not args.no_plots,
    )

    save_dir = Path(metrics.save_dir)
    print(f"\nTest outputs saved to: {save_dir}")
    print(f"mAP50-95: {metrics.box.map:.6f}")
    print(f"mAP50:    {metrics.box.map50:.6f}")
    print(f"mAP75:    {metrics.box.map75:.6f}")


if __name__ == "__main__":
    main()
