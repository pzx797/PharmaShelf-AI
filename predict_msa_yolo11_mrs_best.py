r"""Predict with the best MSA-YOLO11 SKU-to-MRS fine-tuned checkpoint.

Default checkpoint:
    /hy-tmp/ultralytics-main/runs/Fine-tune_MSA-YOLO_sku_to_mrs/MSA_SKU_to_MRS_stage2_seed3/weights/best.pt

Dataset config:
    /hy-tmp/ultralytics-main/MRSDataset.yaml

Default behavior:
    If --source is not specified, the script reads the test split from MRSDataset.yaml
    and predicts on the test images.

Output:
    /hy-tmp/ultralytics-main/runs/predict

Examples:
    python predict_msa_yolo11_sku_to_mrs_seed3.py

    python predict_msa_yolo11_sku_to_mrs_seed3.py --split val

    python predict_msa_yolo11_sku_to_mrs_seed3.py --source /hy-tmp/ultralytics-main/MRSDataset/images/test

    python predict_msa_yolo11_sku_to_mrs_seed3.py --source /hy-tmp/ultralytics-main/MRSDataset/images/test --conf 0.25

    python predict_msa_yolo11_sku_to_mrs_seed3.py --source /hy-tmp/ultralytics-main/MRSDataset/images/test --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(
    "/hy-tmp/ultralytics-main/runs/Fine-tune_MSA-YOLO_sku_to_mrs/MSA_SKU_to_MRS_stage2_seed3/weights/best.pt"
)

DEFAULT_DATA = Path("/hy-tmp/ultralytics-main/MRSDataset.yaml")

DEFAULT_PROJECT = Path("/hy-tmp/ultralytics-main/runs/predict")

DEFAULT_NAME = "MSA_SKU_to_MRS_stage2_seed3_best"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load dataset yaml file."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset yaml not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset yaml format: {path}")

    return data


def resolve_dataset_root(data_cfg: Path, data_dict: dict[str, Any]) -> Path:
    """Resolve dataset root path from YOLO dataset yaml."""
    yaml_dir = data_cfg.parent

    root_value = data_dict.get("path", "")
    if root_value is None:
        root_value = ""

    root_path = Path(str(root_value))

    if root_path.is_absolute():
        return root_path

    if str(root_value).strip():
        return (yaml_dir / root_path).resolve()

    return yaml_dir.resolve()


def resolve_split_source(data_cfg: Path, split: str) -> str:
    """Resolve train/val/test image path from dataset yaml."""
    data_dict = load_yaml(data_cfg)
    dataset_root = resolve_dataset_root(data_cfg, data_dict)

    if split not in data_dict:
        available = [k for k in ["train", "val", "test"] if k in data_dict]
        raise KeyError(f"Split '{split}' not found in {data_cfg}. Available splits: {available}")

    split_value = data_dict[split]

    if isinstance(split_value, list):
        resolved_paths = []
        for item in split_value:
            item_path = Path(str(item))
            if item_path.is_absolute():
                resolved_paths.append(str(item_path))
            else:
                resolved_paths.append(str((dataset_root / item_path).resolve()))
        return resolved_paths[0] if len(resolved_paths) == 1 else "\n".join(resolved_paths)

    split_path = Path(str(split_value))

    if split_path.is_absolute():
        return str(split_path)

    return str((dataset_root / split_path).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with the best MSA-YOLO11 SKU-to-MRS fine-tuned checkpoint.")

    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Image, folder, video, webcam id, or stream URL. "
            "If omitted, the script uses the selected split from MRSDataset.yaml."
        ),
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used when --source is not specified.",
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to .pt model weights.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to MRSDataset.yaml.",
    )

    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--device", default="0", help="CUDA device id, 'cpu', or comma-separated ids.")

    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT,
        help="Output parent directory.",
    )

    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help="Output run name.",
    )

    parser.add_argument(
        "--exist-ok",
        action="store_true",
        default=True,
        help="Overwrite/reuse existing output directory.",
    )

    parser.add_argument(
        "--save-txt",
        action="store_true",
        default=True,
        help="Save YOLO txt labels.",
    )

    parser.add_argument(
        "--no-save-txt",
        action="store_false",
        dest="save_txt",
        help="Do not save txt labels.",
    )

    parser.add_argument(
        "--save-conf",
        action="store_true",
        default=True,
        help="Save confidence in txt labels.",
    )

    parser.add_argument(
        "--no-save-conf",
        action="store_false",
        dest="save_conf",
        help="Do not save confidence in txt labels.",
    )

    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped detections.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show prediction windows.",
    )

    parser.add_argument(
        "--line-width",
        type=int,
        default=None,
        help="Bounding box line width.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = args.weights.resolve()
    data_cfg = args.data.resolve()
    project = args.project.resolve()

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    if not data_cfg.is_file():
        raise FileNotFoundError(f"Dataset yaml not found: {data_cfg}")

    if args.source is None:
        source = resolve_split_source(data_cfg, args.split)
        print(f"No --source specified. Using {args.split} split from dataset yaml:")
        print(f"source = {source}")
    else:
        source = args.source

    model = YOLO(str(weights))

    results = model.predict(
        source=source,
        data=str(data_cfg),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=args.exist_ok,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show=args.show,
        line_width=args.line_width,
    )

    save_dir = Path(results[0].save_dir) if results else project / args.name

    print("\nPrediction completed.")
    print(f"Weights: {weights}")
    print(f"Dataset yaml: {data_cfg}")
    print(f"Source: {source}")
    print(f"Prediction outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
