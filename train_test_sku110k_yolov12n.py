"""
Train and evaluate YOLOv8n, YOLO11n and YOLO26n on SKU-110K.

Workflow
--------
1. Train each model for 300 epochs on SKU110K_fixed.yaml.
2. Load each run's best.pt.
3. Re-evaluate best.pt on the official val split.
4. Evaluate the same best.pt on the official test split.
5. Save a CSV/TXT summary containing validation and test metrics.

Notes:
-----
- The three models are trained sequentially in one script to avoid GPU OOM.
- All models use the same user-specified training arguments.
- Pretrained .pt checkpoints are used as initialization.
"""

# 服务器训练   tmux new -s yolo_train02
# 重新连接 d
# 先按 Ctrl + b，松开，再按 d
# tmux ls

from __future__ import annotations

import csv
import traceback
from pathlib import Path

import torch

import ultralytics
from ultralytics import YOLO

# ============================================================
# 1. Basic configuration
# ============================================================

DATA_CFG = Path("/hy-tmp/ultralytics-main/SKU110K_fixed.yaml")
RUNS_DIR = Path("/hy-tmp/ultralytics-main/runs/SKU_model_compare")

MODELS = {
    "YOLOv12n": "yolo12n.pt",
}

EPOCHS = 300
IMGSZ = 960
BATCH = 16
WORKERS = 0
SEED = 0

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. Shared training arguments
# ============================================================

TRAIN_ARGS = {
    "data": str(DATA_CFG),
    "workers": WORKERS,
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "batch": BATCH,
    "patience": 0,
    # Basic augmentation
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "degrees": 5.0,
    "translate": 0.05,
    "scale": 0.15,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.3,
    # Reproducibility / runtime
    "seed": SEED,
    "deterministic": True,
    "device": DEVICE,
    # Output
    "project": str(RUNS_DIR),
    "plots": True,
    "save": True,
    "verbose": True,
}


# ============================================================
# 3. Helper functions
# ============================================================


def safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_det_metrics(metrics) -> dict[str, float]:
    """Extract object-detection P/R/mAP metrics from Ultralytics metrics object. Compatible with common current
    Ultralytics DetMetrics APIs.
    """
    # Preferred API
    if hasattr(metrics, "box"):
        box = metrics.box
        return {
            "P": safe_float(getattr(box, "mp", None)),
            "R": safe_float(getattr(box, "mr", None)),
            "mAP50": safe_float(getattr(box, "map50", None)),
            "mAP50_95": safe_float(getattr(box, "map", None)),
        }

    # Fallback: results_dict
    rd = getattr(metrics, "results_dict", {}) or {}

    def pick(*keys):
        for k in keys:
            if k in rd:
                return safe_float(rd[k])
        return float("nan")

    return {
        "P": pick("metrics/precision(B)", "metrics/precision"),
        "R": pick("metrics/recall(B)", "metrics/recall"),
        "mAP50": pick("metrics/mAP50(B)", "metrics/mAP50"),
        "mAP50_95": pick("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }


def get_model_complexity(model: YOLO) -> dict[str, float]:
    """Return parameter count plus GFLOPs at 640 and 960 input sizes.

    GFLOPs@640 is useful for keeping the same complexity-reporting convention as many Ultralytics model tables.
    GFLOPs@960 reflects this experiment's actual training/evaluation image size.
    """
    params = sum(p.numel() for p in model.model.parameters())
    gflops_640 = float("nan")
    gflops_960 = float("nan")

    try:
        info_640 = model.info(imgsz=640)
        if info_640 is not None and len(info_640) >= 4:
            params = int(info_640[1])
            gflops_640 = safe_float(info_640[3])
    except Exception:
        pass

    try:
        info_960 = model.info(imgsz=IMGSZ)
        if info_960 is not None and len(info_960) >= 4:
            params = int(info_960[1])
            gflops_960 = safe_float(info_960[3])
    except Exception:
        pass

    return {
        "Params": params,
        "Params_M": params / 1e6,
        "Params_x1e5": params / 1e5,
        "GFLOPs_640": gflops_640,
        "GFLOPs_960": gflops_960,
    }


def read_best_epoch_from_results_csv(results_csv: Path) -> dict[str, float]:
    """Read Ultralytics results.csv and locate the epoch with the highest validation mAP@50:95. This is a reference
    value only.

    The actual reported validation/test metrics are re-evaluated using best.pt.
    """
    result = {
        "best_epoch_by_csv_mAP50_95": float("nan"),
        "csv_best_P": float("nan"),
        "csv_best_R": float("nan"),
        "csv_best_mAP50": float("nan"),
        "csv_best_mAP50_95": float("nan"),
    }

    if not results_csv.exists():
        return result

    with results_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw_row in reader:
            row = {str(k).strip(): v for k, v in raw_row.items()}
            rows.append(row)

    if not rows:
        return result

    def find_key(row: dict, candidates) -> str | None:
        for c in candidates:
            if c in row:
                return c
        # Loose fallback for minor header changes
        lowered = {k.lower().replace(" ", ""): k for k in row}
        for c in candidates:
            n = c.lower().replace(" ", "")
            if n in lowered:
                return lowered[n]
        return None

    sample = rows[0]
    epoch_key = find_key(sample, ["epoch"])
    p_key = find_key(sample, ["metrics/precision(B)", "metrics/precision"])
    r_key = find_key(sample, ["metrics/recall(B)", "metrics/recall"])
    map50_key = find_key(sample, ["metrics/mAP50(B)", "metrics/mAP50"])
    map5095_key = find_key(sample, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])

    if map5095_key is None:
        return result

    valid_rows = []
    for row in rows:
        score = safe_float(row.get(map5095_key))
        if score == score:  # not NaN
            valid_rows.append((score, row))

    if not valid_rows:
        return result

    _, best_row = max(valid_rows, key=lambda x: x[0])

    result["best_epoch_by_csv_mAP50_95"] = safe_float(best_row.get(epoch_key) if epoch_key else None)
    result["csv_best_P"] = safe_float(best_row.get(p_key) if p_key else None)
    result["csv_best_R"] = safe_float(best_row.get(r_key) if r_key else None)
    result["csv_best_mAP50"] = safe_float(best_row.get(map50_key) if map50_key else None)
    result["csv_best_mAP50_95"] = safe_float(best_row.get(map5095_key))

    return result


def write_csv(rows, output_path: Path):
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, digits=4):
    try:
        x = float(x)
        if x != x:
            return "NA"
        return f"{x:.{digits}f}"
    except Exception:
        return str(x)


# ============================================================
# 4. Main
# ============================================================


def main():
    if not DATA_CFG.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_CFG}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("SKU-110K YOLOv12n comparison")
    print(f"Ultralytics version : {ultralytics.__version__}")
    print(f"PyTorch version     : {torch.__version__}")
    print(f"CUDA available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU                 : {torch.cuda.get_device_name(0)}")
    print(f"Device              : {DEVICE}")
    print(f"Dataset             : {DATA_CFG}")
    print(f"Epochs              : {EPOCHS}")
    print(f"Image size          : {IMGSZ}")
    print(f"Batch size          : {BATCH}")
    print(f"Results root        : {RUNS_DIR}")
    print("=" * 90)

    summary_rows = []
    failure_rows = []

    for display_name, weights in MODELS.items():
        run_name = f"SKU110K_{display_name}_300e"

        print("\n" + "#" * 90)
        print(f"START: {display_name}")
        print(f"Initial weights: {weights}")
        print(f"Run name       : {run_name}")
        print("#" * 90)

        try:
            # ----------------------------------------------------
            # A. Train
            # ----------------------------------------------------
            model = YOLO(weights)

            model.train(
                **TRAIN_ARGS,
                name=run_name,
                exist_ok=False,
            )

            train_dir = Path(model.trainer.save_dir)
            best_pt = train_dir / "weights" / "best.pt"
            results_csv = train_dir / "results.csv"

            if not best_pt.exists():
                raise FileNotFoundError(f"best.pt not found after training: {best_pt}")

            csv_best = read_best_epoch_from_results_csv(results_csv)

            # ----------------------------------------------------
            # B. Reload best.pt
            # ----------------------------------------------------
            best_model = YOLO(str(best_pt))
            complexity = get_model_complexity(best_model)

            # ----------------------------------------------------
            # C. Validate best.pt on official validation split
            #    This is used as the final "training/validation result".
            # ----------------------------------------------------
            val_name = f"{run_name}_best_VAL"
            val_metrics_obj = best_model.val(
                data=str(DATA_CFG),
                split="val",
                imgsz=IMGSZ,
                batch=BATCH,
                workers=WORKERS,
                device=DEVICE,
                project=str(RUNS_DIR),
                name=val_name,
                plots=True,
                verbose=True,
            )
            val_metrics = extract_det_metrics(val_metrics_obj)

            # ----------------------------------------------------
            # D. Test best.pt on official test split
            # ----------------------------------------------------
            test_name = f"{run_name}_best_TEST"
            test_metrics_obj = best_model.val(
                data=str(DATA_CFG),
                split="test",
                imgsz=IMGSZ,
                batch=BATCH,
                workers=WORKERS,
                device=DEVICE,
                project=str(RUNS_DIR),
                name=test_name,
                plots=True,
                verbose=True,
            )
            test_metrics = extract_det_metrics(test_metrics_obj)

            row = {
                "Model": display_name,
                "InitialWeights": weights,
                "Epochs": EPOCHS,
                "ImgSz": IMGSZ,
                "Batch": BATCH,
                "Seed": SEED,
                "BestEpoch_by_resultsCSV_mAP50_95": csv_best["best_epoch_by_csv_mAP50_95"],
                # best.pt re-evaluated on official val
                "Val_P": val_metrics["P"],
                "Val_R": val_metrics["R"],
                "Val_mAP50": val_metrics["mAP50"],
                "Val_mAP50_95": val_metrics["mAP50_95"],
                # same best.pt evaluated on official test
                "Test_P": test_metrics["P"],
                "Test_R": test_metrics["R"],
                "Test_mAP50": test_metrics["mAP50"],
                "Test_mAP50_95": test_metrics["mAP50_95"],
                "Params": complexity["Params"],
                "Params_M": complexity["Params_M"],
                "Params_x1e5": complexity["Params_x1e5"],
                "GFLOPs_at_640": complexity["GFLOPs_640"],
                "GFLOPs_at_960": complexity["GFLOPs_960"],
                "TrainDir": str(train_dir),
                "BestPt": str(best_pt),
                "ValDir": str(RUNS_DIR / val_name),
                "TestDir": str(RUNS_DIR / test_name),
            }
            summary_rows.append(row)

            # Save/update summary after each model, so partial results survive
            # even if a later model fails.
            summary_csv = RUNS_DIR / "SKU110K_YOLOv12n_train_test_summary.csv"
            write_csv(summary_rows, summary_csv)

            print("\n" + "-" * 90)
            print(f"{display_name} finished")
            print(
                "VAL : "
                f"P={fmt(val_metrics['P'])}, "
                f"R={fmt(val_metrics['R'])}, "
                f"mAP50={fmt(val_metrics['mAP50'])}, "
                f"mAP50-95={fmt(val_metrics['mAP50_95'])}"
            )
            print(
                "TEST: "
                f"P={fmt(test_metrics['P'])}, "
                f"R={fmt(test_metrics['R'])}, "
                f"mAP50={fmt(test_metrics['mAP50'])}, "
                f"mAP50-95={fmt(test_metrics['mAP50_95'])}"
            )
            print(
                f"Params={complexity['Params']:,} "
                f"({complexity['Params_x1e5']:.2f}×10^5), "
                f"GFLOPs@640={fmt(complexity['GFLOPs_640'], 2)}, "
                f"GFLOPs@960={fmt(complexity['GFLOPs_960'], 2)}"
            )
            print(f"best.pt: {best_pt}")
            print("-" * 90)

            # Free GPU memory before next model
            del model
            del best_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[ERROR] {display_name} failed: {e}")
            traceback.print_exc()

            failure_rows.append(
                {
                    "Model": display_name,
                    "Weights": weights,
                    "Error": repr(e),
                }
            )
            write_csv(
                failure_rows,
                RUNS_DIR / "SKU110K_YOLOv12n_failures.csv",
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========================================================
    # 5. Final summary
    # ========================================================
    summary_csv = RUNS_DIR / "SKU110K_YOLOv12n_train_test_summary.csv"
    summary_txt = RUNS_DIR / "SKU110K_YOLOv12n_train_test_summary.txt"

    write_csv(summary_rows, summary_csv)

    lines = []
    lines.append("=" * 120)
    lines.append("SKU-110K FINAL SUMMARY")
    lines.append("=" * 120)

    header = (
        f"{'Model':<12}"
        f"{'Val P':>9}{'Val R':>9}{'Val mAP50':>12}{'Val mAP50-95':>15}"
        f"{'Test P':>10}{'Test R':>10}{'Test mAP50':>13}{'Test mAP50-95':>16}"
        f"{'Params(1e5)':>14}{'GFLOPs@640':>13}{'GFLOPs@960':>13}"
    )
    lines.append(header)
    lines.append("-" * 120)

    for row in summary_rows:
        lines.append(
            f"{row['Model']:<12}"
            f"{fmt(row['Val_P']):>9}"
            f"{fmt(row['Val_R']):>9}"
            f"{fmt(row['Val_mAP50']):>12}"
            f"{fmt(row['Val_mAP50_95']):>15}"
            f"{fmt(row['Test_P']):>10}"
            f"{fmt(row['Test_R']):>10}"
            f"{fmt(row['Test_mAP50']):>13}"
            f"{fmt(row['Test_mAP50_95']):>16}"
            f"{fmt(row['Params_x1e5'], 2):>14}"
            f"{fmt(row['GFLOPs_at_640'], 2):>13}"
            f"{fmt(row['GFLOPs_at_960'], 2):>13}"
        )

    lines.append("=" * 120)
    lines.append(f"CSV summary: {summary_csv}")

    if failure_rows:
        lines.append("")
        lines.append("FAILED MODELS:")
        for item in failure_rows:
            lines.append(f"- {item['Model']}: {item['Error']}")
        lines.append(f"Failure log: {RUNS_DIR / 'SKU110K_YOLOv12n_failures.csv'}")

    text = "\n".join(lines)
    summary_txt.write_text(text, encoding="utf-8")

    print("\n" + text)
    print(f"\nTXT summary: {summary_txt}")


if __name__ == "__main__":
    main()
