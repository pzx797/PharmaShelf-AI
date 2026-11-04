"""Run two trained YOLO models and save images with bounding boxes only."""

from argparse import ArgumentParser
from pathlib import Path

import yaml

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "MRSDataset.yaml"
DEFAULT_MODELS = (
    ROOT
    / "runs/Fine-tune_MSA-YOLO_sku_to_mrs/"
    "MSA_SKU_to_MRS_stage2_seed3/weights/best.pt",
    Path(r"E:\CODE\ultralytics-main\runs\multiseed_baseline")
    / "baseline_yolo11n_seed3/weights/best.pt",
)


def dataset_split_path(data_yaml: Path, split: str) -> Path:
    """Resolve a train/val/test image path from an Ultralytics dataset YAML."""
    with data_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    split_value = data.get(split)
    if not split_value:
        raise KeyError(f"{data_yaml} 中没有配置 {split!r} 数据集")
    if isinstance(split_value, list):
        raise ValueError("默认输入不支持多个图片目录，请通过 --source 显式指定一个目录")

    dataset_root = Path(data.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    source = Path(split_value)
    return source if source.is_absolute() else dataset_root / source


def parse_args():
    parser = ArgumentParser(description="使用两个 YOLO 模型预测，并只绘制预测框")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="数据集 YAML")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--source", type=Path, help="图片、图片目录或视频；不传则使用 YAML 的 split")
    parser.add_argument("--model1", type=Path, default=DEFAULT_MODELS[0])
    parser.add_argument("--model2", type=Path, default=DEFAULT_MODELS[1])
    parser.add_argument("--project", type=Path, default=ROOT / "runs/predict_boxes_only")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None, help="例如 0、0,1 或 cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source or dataset_split_path(args.data.resolve(), args.split)
    if not source.exists():
        raise FileNotFoundError(f"预测输入不存在：{source}")

    for name, weight in (("msa_yolo", args.model1), ("baseline_yolo11n", args.model2)):
        if not weight.exists():
            raise FileNotFoundError(f"模型权重不存在：{weight}")

        print(f"\n正在预测：{name}\n权重：{weight}\n输入：{source}")
        YOLO(str(weight)).predict(
            source=str(source),
            project=str(args.project),
            name=name,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save=True,
            save_txt=False,
            save_conf=False,
            show_labels=False,
            show_conf=False,
            exist_ok=True,
        )

    print(f"\n预测完成，结果保存在：{args.project.resolve()}")


if __name__ == "__main__":
    main()
