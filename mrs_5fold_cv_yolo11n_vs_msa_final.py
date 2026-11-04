#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRS 5-fold cross-validation for YOLO11n vs MSA-YOLO11.

Workflow:
- Pool original train/val/test images from MRSDataset.yaml.
- Deterministic 5-fold split on all images (seed=1 by default).
- For each fold: held-out test fold + inner train/val split.
- Train YOLO11n and MSA-YOLO11 on exactly the same fold.
- Use fixed seed for all folds/models.
- Two-stage fine-tuning: 50 epochs freeze=11 + 250 epochs full training.
- Evaluate final best.pt on held-out test.
- Save per-fold metrics and Mean±SD + 95% CI.
"""

# 服务器训练   tmux new -s yolo_train
# 重新连接 tmux attach -t yolo_train
# 先按 Ctrl + b，松开，再按 d
# tmux ls

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import statistics
from pathlib import Path
from typing import List, Sequence, Tuple

import yaml
from sklearn.model_selection import KFold, train_test_split
from ultralytics import YOLO

PROJECT_ROOT = Path("/hy-tmp/ultralytics-main")
ORIGINAL_DATA_YAML = PROJECT_ROOT / "MRSDataset.yaml"
MSA_MODEL_CFG = PROJECT_ROOT / "yolo11n-msa.yaml"

YOLO11N_SKU_WEIGHTS = PROJECT_ROOT / "runs/train15/weights/best.pt"
MSA_SKU_WEIGHTS = PROJECT_ROOT / "runs/MSA-YOLO_mrs_pretrained/yolo11n_MSA_SKU_final/weights/best.pt"

CV_SPLIT_ROOT = PROJECT_ROOT / "datasets/MRSDataset_5fold"
RUN_PROJECT = PROJECT_ROOT / "runs/MRS_5fold_CV"

EXPECTED_IMAGE_COUNT = 99
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

N_SPLITS = 5
DEFAULT_SEED = 1
INNER_VAL_RATIO = 0.125

STAGE1_EPOCHS = 50
STAGE2_EPOCHS = 250

COMMON_TRAIN_ARGS = dict(
    workers=0,
    imgsz=960,
    batch=16,
    device=0,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_bias_lr=0.01,
    cos_lr=False,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,
    degrees=5.0,
    translate=0.05,
    scale=0.15,
    shear=0.0,
    perspective=0.0,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.3,
    close_mosaic=50,
    patience=0,
    pretrained=False,
    deterministic=True,
    plots=True,
    save=True,
    exist_ok=True,
)


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]


def _resolve_dataset_root(cfg: dict, yaml_path: Path) -> Path:
    raw_root = cfg.get("path")
    if raw_root is None:
        return yaml_path.parent.resolve()
    root = Path(str(raw_root))
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def _resolve_split_source(source: str, dataset_root: Path) -> Path:
    p = Path(source)
    if not p.is_absolute():
        p = dataset_root / p
    return p.resolve()


def _read_image_txt(txt_path: Path, dataset_root: Path) -> List[Path]:
    images = []
    with txt_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            p = Path(raw)
            if not p.is_absolute():
                p1 = (dataset_root / p).resolve()
                p2 = (txt_path.parent / p).resolve()
                p = p1 if p1.exists() else p2
            images.append(p.resolve())
    return images


def _collect_images_from_source(source: str, dataset_root: Path) -> List[Path]:
    p = _resolve_split_source(source, dataset_root)

    if p.is_dir():
        return sorted(
            x.resolve()
            for x in p.rglob("*")
            if x.is_file() and x.suffix.lower() in IMAGE_SUFFIXES
        )

    if p.is_file() and p.suffix.lower() == ".txt":
        return _read_image_txt(p, dataset_root)

    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
        return [p.resolve()]

    raise FileNotFoundError(f"Cannot resolve dataset source: {source}\nResolved: {p}")


def collect_all_images(data_yaml: Path) -> Tuple[List[Path], dict]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    dataset_root = _resolve_dataset_root(cfg, data_yaml)

    images: List[Path] = []
    for split_name in ("train", "val", "test"):
        for source in _as_list(cfg.get(split_name)):
            images.extend(_collect_images_from_source(source, dataset_root))

    unique = sorted({p.resolve() for p in images})

    missing = [p for p in unique if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Some images do not exist:\n" + "\n".join(str(x) for x in missing[:20])
        )

    return unique, cfg


def image_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    indices = [i for i, part in enumerate(parts) if part == "images"]
    if not indices:
        raise RuntimeError(
            f"Cannot infer label path because 'images' is not in:\n{image_path}"
        )
    idx = indices[-1]
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def validate_labels(images: Sequence[Path]) -> None:
    missing_labels = []
    for img in images:
        label = image_to_label_path(img)
        if not label.is_file():
            missing_labels.append((img, label))

    if missing_labels:
        examples = "\n".join(
            f"image: {img}\nlabel: {label}"
            for img, label in missing_labels[:10]
        )
        raise FileNotFoundError(
            f"{len(missing_labels)} images have no matching label.\n{examples}"
        )


def write_path_list(paths: Sequence[Path], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p.resolve()) + "\n")


def create_fold_yaml(
    fold_dir: Path,
    original_cfg: dict,
    train_paths: Sequence[Path],
    val_paths: Sequence[Path],
    test_paths: Sequence[Path],
) -> Path:
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_txt = fold_dir / "train.txt"
    val_txt = fold_dir / "val.txt"
    test_txt = fold_dir / "test.txt"

    write_path_list(train_paths, train_txt)
    write_path_list(val_paths, val_txt)
    write_path_list(test_paths, test_txt)

    fold_cfg = {
        "path": str(_resolve_dataset_root(original_cfg, ORIGINAL_DATA_YAML)),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "test": str(test_txt.resolve()),
    }
    if "names" in original_cfg:
        fold_cfg["names"] = original_cfg["names"]
    if "nc" in original_cfg:
        fold_cfg["nc"] = original_cfg["nc"]

    out_yaml = fold_dir / "dataset.yaml"
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(fold_cfg, f, allow_unicode=True, sort_keys=False)

    return out_yaml


def build_five_folds(
    all_images: Sequence[Path],
    original_cfg: dict,
    seed: int,
    reset_splits: bool = False,
) -> List[Path]:
    if reset_splits and CV_SPLIT_ROOT.exists():
        shutil.rmtree(CV_SPLIT_ROOT)

    CV_SPLIT_ROOT.mkdir(parents=True, exist_ok=True)

    images = list(all_images)
    indices = list(range(len(images)))

    outer = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fold_yamls: List[Path] = []

    for fold_no, (trainval_idx, test_idx) in enumerate(outer.split(indices), start=1):
        trainval_paths = [images[i] for i in trainval_idx]
        test_paths = [images[i] for i in test_idx]

        train_paths, val_paths = train_test_split(
            trainval_paths,
            test_size=INNER_VAL_RATIO,
            random_state=seed,
            shuffle=True,
        )

        fold_dir = CV_SPLIT_ROOT / f"fold_{fold_no}"
        fold_yaml = create_fold_yaml(
            fold_dir,
            original_cfg,
            sorted(train_paths),
            sorted(val_paths),
            sorted(test_paths),
        )
        fold_yamls.append(fold_yaml)

        print(
            f"[Fold {fold_no}] train={len(train_paths)}, "
            f"val={len(val_paths)}, test={len(test_paths)}"
        )

    return fold_yamls


def make_yolo11n_from_sku_checkpoint(weights: Path) -> YOLO:
    if not weights.is_file():
        raise FileNotFoundError(
            f"YOLO11n SKU-110K checkpoint not found:\n{weights}\n"
            "Use --yolo11n-weights /path/to/best.pt"
        )
    return YOLO(str(weights))


def make_msa_from_sku_checkpoint(weights: Path) -> YOLO:
    if not MSA_MODEL_CFG.is_file():
        raise FileNotFoundError(f"MSA YAML not found: {MSA_MODEL_CFG}")
    if not weights.is_file():
        raise FileNotFoundError(f"MSA SKU checkpoint not found: {weights}")

    model = YOLO(str(MSA_MODEL_CFG))
    model.load(str(weights))
    return model


def extract_detection_metrics(metrics) -> dict:
    d = metrics.results_dict
    return {
        "precision": float(d.get("metrics/precision(B)", 0.0)),
        "recall": float(d.get("metrics/recall(B)", 0.0)),
        "map50": float(d.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(d.get("metrics/mAP50-95(B)", 0.0)),
    }


def train_one_fold(
    model_name: str,
    fold_no: int,
    fold_yaml: Path,
    seed: int,
    yolo11n_weights: Path,
    msa_weights: Path,
) -> dict:
    print("\n" + "=" * 90)
    print(f"{model_name} | Fold {fold_no}/{N_SPLITS}")
    print(f"data={fold_yaml}")
    print(f"seed={seed}")
    print("=" * 90)

    if model_name == "YOLO11n":
        model = make_yolo11n_from_sku_checkpoint(yolo11n_weights)
    elif model_name == "MSA-YOLO11":
        model = make_msa_from_sku_checkpoint(msa_weights)
    else:
        raise ValueError(model_name)

    stage1_args = {
        **COMMON_TRAIN_ARGS,
        "data": str(fold_yaml),
        "project": str(RUN_PROJECT),
        "name": f"{model_name}_fold{fold_no}_stage1",
        "epochs": STAGE1_EPOCHS,
        "freeze": 11,
    }
    model.train(**stage1_args, seed=seed)

    stage1_best = Path(model.trainer.best)
    if not stage1_best.is_file():
        raise FileNotFoundError(f"Stage1 best.pt not found: {stage1_best}")

    model = YOLO(str(stage1_best))

    stage2_args = {
        **COMMON_TRAIN_ARGS,
        "data": str(fold_yaml),
        "project": str(RUN_PROJECT),
        "name": f"{model_name}_fold{fold_no}_stage2",
        "epochs": STAGE2_EPOCHS,
        "freeze": None,
        "lr0": 0.006,
        "close_mosaic": 80,
    }
    model.train(**stage2_args, seed=seed)

    stage2_best = Path(model.trainer.best)
    if not stage2_best.is_file():
        raise FileNotFoundError(f"Stage2 best.pt not found: {stage2_best}")

    final_model = YOLO(str(stage2_best))
    metrics = final_model.val(
        data=str(fold_yaml),
        split="test",
        imgsz=960,
        batch=16,
        workers=0,
        device=0,
        plots=True,
        project=str(RUN_PROJECT),
        name=f"{model_name}_fold{fold_no}_TEST",
    )

    result = extract_detection_metrics(metrics)
    result.update(
        {
            "model": model_name,
            "fold": fold_no,
            "seed": seed,
            "train_yaml": str(fold_yaml),
            "best_pt": str(stage2_best),
        }
    )

    print(
        f"[TEST] {model_name} fold={fold_no}: "
        f"P={result['precision']:.6f}, "
        f"R={result['recall']:.6f}, "
        f"mAP50={result['map50']:.6f}, "
        f"mAP50-95={result['map50_95']:.6f}"
    )
    return result


RESULT_FIELDS = [
    "model", "fold", "seed", "precision", "recall",
    "map50", "map50_95", "train_yaml", "best_pt"
]


def save_fold_results(rows: Sequence[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd_ci95(values: Sequence[float]) -> Tuple[float, float, float, float]:
    vals = list(map(float, values))
    n = len(vals)
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    t_crit = 2.7764451051977987 if n == 5 else 1.959963984540054
    margin = t_crit * sd / math.sqrt(n) if n > 1 else 0.0
    return mean, sd, mean - margin, mean + margin


def build_summary(rows: Sequence[dict]) -> List[dict]:
    summary_rows = []
    metrics = ["precision", "recall", "map50", "map50_95"]

    for model_name in ("YOLO11n", "MSA-YOLO11"):
        model_rows = [r for r in rows if r["model"] == model_name]
        if len(model_rows) != N_SPLITS:
            raise RuntimeError(
                f"{model_name}: expected {N_SPLITS} folds, got {len(model_rows)}"
            )

        for metric in metrics:
            values = [float(r[metric]) for r in model_rows]
            mean, sd, ci_low, ci_high = mean_sd_ci95(values)
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "mean_plus_minus_sd": f"{mean:.6f} ± {sd:.6f}",
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "ci95": f"[{ci_low:.6f}, {ci_high:.6f}]",
                }
            )
    return summary_rows


def save_summary(rows: Sequence[dict], out_csv: Path) -> None:
    fields = [
        "model", "metric", "mean", "sd", "mean_plus_minus_sd",
        "ci95_low", "ci95_high", "ci95"
    ]
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows: Sequence[dict]) -> None:
    pretty = {
        "precision": "Precision",
        "recall": "Recall",
        "map50": "mAP@50",
        "map50_95": "mAP@50:95",
    }
    print("\n" + "=" * 90)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 90)

    for model_name in ("YOLO11n", "MSA-YOLO11"):
        print(f"\n{model_name}")
        for row in summary_rows:
            if row["model"] == model_name:
                print(
                    f"  {pretty[row['metric']]:<10} "
                    f"{row['mean_plus_minus_sd']}  "
                    f"95% CI {row['ci95']}"
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="5-fold CV: YOLO11n vs MSA-YOLO11 on MRSDataset"
    )
    parser.add_argument(
        "--yolo11n-weights",
        type=Path,
        default=YOLO11N_SKU_WEIGHTS,
        help=f"YOLO11n SKU-110K pretrained best.pt (default: {YOLO11N_SKU_WEIGHTS})",
    )
    parser.add_argument(
        "--msa-weights",
        type=Path,
        default=MSA_SKU_WEIGHTS,
        help=f"MSA SKU-110K pretrained best.pt (default: {MSA_SKU_WEIGHTS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Fixed seed for data split and training. Default: 1",
    )
    parser.add_argument(
        "--reset-splits",
        action="store_true",
        help="Rebuild fold TXT/YAML definitions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not ORIGINAL_DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {ORIGINAL_DATA_YAML}")
    if not MSA_MODEL_CFG.is_file():
        raise FileNotFoundError(f"MSA YAML not found: {MSA_MODEL_CFG}")
    if not args.yolo11n_weights.is_file():
        raise FileNotFoundError(
            f"YOLO11n SKU checkpoint not found: {args.yolo11n_weights}"
        )
    if not args.msa_weights.is_file():
        raise FileNotFoundError(
            f"MSA SKU checkpoint not found: {args.msa_weights}"
        )

    random.seed(args.seed)

    all_images, original_cfg = collect_all_images(ORIGINAL_DATA_YAML)
    print(f"\nTotal unique images found: {len(all_images)}")

    if len(all_images) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGE_COUNT} images, found {len(all_images)}."
        )

    validate_labels(all_images)
    print("All image-label pairs passed validation.")

    fold_yamls = build_five_folds(
        all_images,
        original_cfg,
        seed=args.seed,
        reset_splits=args.reset_splits,
    )

    RUN_PROJECT.mkdir(parents=True, exist_ok=True)

    manifest_csv = CV_SPLIT_ROOT / "all_images_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image", "label"])
        for idx, img in enumerate(all_images):
            writer.writerow([idx, str(img), str(image_to_label_path(img))])

    all_results: List[dict] = []

    for model_name in ("YOLO11n", "MSA-YOLO11"):
        for fold_no, fold_yaml in enumerate(fold_yamls, start=1):
            result = train_one_fold(
                model_name=model_name,
                fold_no=fold_no,
                fold_yaml=fold_yaml,
                seed=args.seed,
                yolo11n_weights=args.yolo11n_weights,
                msa_weights=args.msa_weights,
            )
            all_results.append(result)
            save_fold_results(
                all_results,
                RUN_PROJECT / "fold_results_partial.csv",
            )

    fold_results_csv = RUN_PROJECT / "fold_results.csv"
    save_fold_results(all_results, fold_results_csv)

    summary_rows = build_summary(all_results)
    summary_csv = RUN_PROJECT / "cross_validation_summary.csv"
    save_summary(summary_rows, summary_csv)

    print_summary(summary_rows)

    print("\nSaved:")
    print(f"  Fold definitions : {CV_SPLIT_ROOT}")
    print(f"  Raw fold results : {fold_results_csv}")
    print(f"  Summary          : {summary_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()
