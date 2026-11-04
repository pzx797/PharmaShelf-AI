"""
Evaluate the 5 directly trained MSA-YOLO MRS runs on the MRSDataset test split.

This script loads best.pt from:
    runs/MSA-YOLO_mrs_seed5/yolo11n_SOSE_MAFF_precision_seed{0..4}/weights/best.pt

It saves per-seed test plots plus a CSV/TXT summary back to:
    runs/MSA-YOLO_mrs_seed5
"""

import csv
from pathlib import Path
from statistics import mean, stdev

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CFG = PROJECT_ROOT / "MRSDataset.yaml"
SAVE_ROOT = PROJECT_ROOT / "runs" / "MSA-YOLO_mrs_seed5"

SEEDS = [0, 1, 2, 3, 4]
RUN_NAME_TEMPLATE = "yolo11n_SOSE_MAFF_precision_seed{seed}"
TEST_NAME_TEMPLATE = "yolo11n_SOSE_MAFF_precision_test_seed{seed}"


COMMON_TEST_ARGS = {
    "data": str(DATA_CFG),
    "split": "test",
    "workers": 0,
    "imgsz": 960,
    "batch": 16,
    "device": 0,
    "plots": True,
    "project": str(SAVE_ROOT),
}


def extract_metrics(metrics):
    """Extract P, R, mAP50, and mAP50-95 from Ultralytics validation results."""
    result_dict = getattr(metrics, "results_dict", {}) or {}

    precision = result_dict.get("metrics/precision(B)", None)
    recall = result_dict.get("metrics/recall(B)", None)
    map50 = result_dict.get("metrics/mAP50(B)", None)
    map50_95 = result_dict.get("metrics/mAP50-95(B)", None)

    if precision is None:
        precision = float(metrics.box.mp)
    if recall is None:
        recall = float(metrics.box.mr)
    if map50 is None:
        map50 = float(metrics.box.map50)
    if map50_95 is None:
        map50_95 = float(metrics.box.map)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "map50": float(map50),
        "map50_95": float(map50_95),
    }


def metric_mean_std(results, key):
    values = [r[key] for r in results]
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def save_summary(results):
    csv_path = SAVE_ROOT / "MSA_MRS_direct_5seeds_test_results.csv"
    txt_path = SAVE_ROOT / "MSA_MRS_direct_5seeds_test_summary.txt"

    p_mean, p_std = metric_mean_std(results, "precision")
    r_mean, r_std = metric_mean_std(results, "recall")
    map50_mean, map50_std = metric_mean_std(results, "map50")
    map50_95_mean, map50_95_std = metric_mean_std(results, "map50_95")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "weight",
                "test_save_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MSA-YOLO11 direct MRSDataset training -> test evaluation\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset config:\n{DATA_CFG}\n\n")
        f.write(f"Save root:\n{SAVE_ROOT}\n\n")

        f.write("Per-seed test results:\n")
        f.writelines(
            f"seed={r['seed']}, "
            f"P={r['precision']:.6f}, "
            f"R={r['recall']:.6f}, "
            f"mAP50={r['map50']:.6f}, "
            f"mAP50-95={r['map50_95']:.6f}\n"
            for r in results
        )

        f.write("\nMean +/- Std over 5 seeds:\n")
        f.write(f"Precision = {p_mean:.6f} +/- {p_std:.6f}\n")
        f.write(f"Recall = {r_mean:.6f} +/- {r_std:.6f}\n")
        f.write(f"mAP50 = {map50_mean:.6f} +/- {map50_std:.6f}\n")
        f.write(f"mAP50-95 = {map50_95_mean:.6f} +/- {map50_95_std:.6f}\n")

    print("\n" + "=" * 80)
    print("Final 5-seed test summary")
    print("=" * 80)
    print(f"Precision = {p_mean:.6f} +/- {p_std:.6f}")
    print(f"Recall = {r_mean:.6f} +/- {r_std:.6f}")
    print(f"mAP50 = {map50_mean:.6f} +/- {map50_std:.6f}")
    print(f"mAP50-95 = {map50_95_mean:.6f} +/- {map50_95_std:.6f}")
    print(f"\nCSV saved to: {csv_path}")
    print(f"Summary saved to: {txt_path}")


def main():
    if not DATA_CFG.is_file():
        raise FileNotFoundError(f"MRSDataset.yaml was not found: {DATA_CFG}")

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = []

    for seed in SEEDS:
        run_dir = SAVE_ROOT / RUN_NAME_TEMPLATE.format(seed=seed)
        weight = run_dir / "weights" / "best.pt"

        if not weight.is_file():
            raise FileNotFoundError(f"best.pt was not found for seed={seed}: {weight}")

        test_name = TEST_NAME_TEMPLATE.format(seed=seed)
        test_save_dir = SAVE_ROOT / test_name

        print("\n" + "=" * 80)
        print(f"Testing direct MRS model on test split: seed={seed}")
        print(f"Weight: {weight}")
        print("=" * 80)

        model = YOLO(str(weight))
        metrics = model.val(
            **COMMON_TEST_ARGS,
            name=test_name,
            exist_ok=True,
        )

        metric_values = extract_metrics(metrics)
        one_result = {
            "seed": seed,
            "precision": metric_values["precision"],
            "recall": metric_values["recall"],
            "map50": metric_values["map50"],
            "map50_95": metric_values["map50_95"],
            "weight": str(weight),
            "test_save_dir": str(test_save_dir),
        }
        all_results.append(one_result)

        print(
            f"\nSeed {seed} test results: "
            f"P={one_result['precision']:.6f}, "
            f"R={one_result['recall']:.6f}, "
            f"mAP50={one_result['map50']:.6f}, "
            f"mAP50-95={one_result['map50_95']:.6f}"
        )

    save_summary(all_results)


if __name__ == "__main__":
    main()
