r"""Run prediction with the trained MRS YOLO detector.

By default, this script predicts on MRSDataset's test image directory and saves
annotated images plus YOLO txt labels under runs/detect/mrs_predict_*.

Example:
    python predict_mrs_yolo11.py
    python predict_mrs_yolo11.py --source D:\images\demo --conf 0.25 --save-crop
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "yolo11n_SOSE_MAFF_precision_seed0" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "MRSDataset.yaml"


def dataset_source_from_yaml(data_yaml: Path, split: str = "test") -> Path:
    """Return an absolute image source path for a split declared in a YOLO dataset YAML."""
    with data_yaml.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    if split not in data:
        raise KeyError(f"Split '{split}' was not found in {data_yaml}")

    root = Path(data.get("path", data_yaml.parent))
    source = Path(data[split])
    return source if source.is_absolute() else root / source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict images/videos with a trained MRS YOLO detector.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to trained .pt weights.")
    parser.add_argument("--source", type=Path, default=None, help="Image, directory, video, or stream source.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML used when --source is omitted.")
    parser.add_argument(
        "--split", default="test", choices=("train", "val", "test"), help="Dataset split used by default."
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold for displayed/saved predictions."
    )
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--device", default="0", help="CUDA device id, 'cpu', or comma-separated device ids.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "detect", help="Output project directory.")
    parser.add_argument("--name", default="mrs_predict_yolo11n_SOSE_MAFF_precision_seed0", help="Output run name.")
    parser.add_argument("--save-txt", action="store_true", default=True, help="Save predictions as YOLO txt labels.")
    parser.add_argument("--no-save-txt", action="store_false", dest="save_txt", help="Do not save YOLO txt labels.")
    parser.add_argument(
        "--save-conf", action="store_true", default=True, help="Include confidence values in txt labels."
    )
    parser.add_argument(
        "--no-save-conf", action="store_false", dest="save_conf", help="Do not include confidence values."
    )
    parser.add_argument("--save-crop", action="store_true", help="Save cropped detections.")
    parser.add_argument("--show", action="store_true", help="Display prediction windows.")
    parser.add_argument("--line-width", type=int, default=None, help="Annotation line width.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    if args.source is None:
        data = args.data.resolve()
        if not data.is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {data}")
        source = dataset_source_from_yaml(data, args.split)
    else:
        source = args.source

    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        project=str(args.project),
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show=args.show,
        line_width=args.line_width,
    )

    save_dir = Path(results[0].save_dir) if results else args.project / args.name
    print(f"\nPrediction outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
