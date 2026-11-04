"""Plot mAP curves for YOLO model comparison (first 100 epochs)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

EPOCHS = 100
OUTPUT_DIR = Path(__file__).resolve().parent / "map_curve_plots"

RESULT_FILES = {
    "YOLO26n": Path(r"E:\CODE\ultralytics-main\runs\detect\train9\results.csv"),
    "YOLOv12n": Path(r"E:\Desktop\model_compare\yolov12\runs\train\mrs_yolov12n\results.csv"),
    "YOLOv8n": Path(r"E:\CODE\ultralytics-main\runs\detect\train11\results.csv"),
    "YOLO11n": Path(r"E:\CODE\ultralytics-main\runs\multiseed_baseline\baseline_yolo11n_seed04\results.csv"),
    "YOLO11n-MobileNetV3": Path(
        r"E:\Desktop\model_compare\ultralytics-main01"
        r"\runs\yolo11n_mobilenetv3\results.csv"
    ),
    "YOLOv5n": Path(r"E:\CODE\ultralytics-main\runs\detect\train12\results.csv"),
    "MDI-YOLO": Path(r"E:\Desktop\model_compare\ultralytics-main\runs\train\mrs_yolov8\results.csv"),
}

MSA_STAGE1 = Path(
    r"E:\EdgeDownload\ultralytics-main\runs\MSA-YOLO_mrs_seed5"
    r"\yolo11n_SOSE_MAFF_precision_stage1_seed1\results.csv"
)
MSA_STAGE2 = Path(
    r"E:\EdgeDownload\ultralytics-main\runs\MSA-YOLO_mrs_seed5"
    r"\yolo11n_SOSE_MAFF_precision_seed1\results.csv"
)

COLORS = {
    "YOLO26n": "#1f77b4",
    "YOLOv12n": "#ff7f0e",
    "YOLOv8n": "#2ca02c",
    "YOLO11n": "#d62728",
    "YOLO11n-MobileNetV3": "#17becf",
    "YOLOv5n": "#9467bd",
    "MDI-YOLO": "#8c564b",
    "MSA-YOLO (Ours)": "#e31a1c",
}

METRICS = {
    "metrics/mAP50(B)": ("mAP@50 vs. Epoch", "mAP@50", "map50_curve.png"),
    "metrics/mAP50-95(B)": (
        "mAP@50:95 vs. Epoch",
        "mAP@50:95",
        "map50_95_curve.png",
    ),
}


def read_results(path: Path) -> pd.DataFrame:
    """Read a normal result file, deduplicate epochs, and select epochs 1..100."""
    if not path.is_file():
        raise FileNotFoundError(f"找不到训练结果文件：{path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = {"epoch", *METRICS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少列：{sorted(missing)}")

    # Some resumed CSVs contain repeated epoch blocks. Keep the latest record.
    df = df.drop_duplicates(subset="epoch", keep="last").sort_values("epoch")
    df = df[df["epoch"].between(1, EPOCHS)].copy()
    if len(df) < EPOCHS:
        raise ValueError(f"{path} 只有 {len(df)} 个可用 epoch，少于要求的 {EPOCHS} 个")
    return df.iloc[:EPOCHS]


def read_msa_results(stage1_path: Path, stage2_path: Path) -> pd.DataFrame:
    """Join MSA stage 1 (epochs 1..50) and stage 2 (epochs 51..100)."""
    for path in (stage1_path, stage2_path):
        if not path.is_file():
            raise FileNotFoundError(f"找不到 MSA-YOLO 训练结果文件：{path}")

    stage1 = pd.read_csv(stage1_path)
    stage2 = pd.read_csv(stage2_path)
    stage1.columns = stage1.columns.str.strip()
    stage2.columns = stage2.columns.str.strip()

    required = {"epoch", *METRICS}
    for path, df in ((stage1_path, stage1), (stage2_path, stage2)):
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} 缺少列：{sorted(missing)}")

    first = stage1.iloc[:50].copy()
    second = stage2.iloc[: EPOCHS - len(first)].copy()
    msa = pd.concat([first, second], ignore_index=True)
    if len(msa) < EPOCHS:
        raise ValueError(f"MSA-YOLO 拼接后只有 {len(msa)} 个 epoch")

    # Ignore each source file's local epoch values and assign global epochs.
    msa["epoch"] = range(1, EPOCHS + 1)
    return msa


def plot_metric(data: dict[str, pd.DataFrame], metric: str) -> Path:
    title, ylabel, filename = METRICS[metric]
    fig, ax = plt.subplots(figsize=(11, 7), dpi=160)

    for name, df in data.items():
        is_ours = name == "MSA-YOLO (Ours)"
        ax.plot(
            df["epoch"],
            df[metric],
            label=name,
            color=COLORS[name],
            linewidth=2.5 if is_ours else 1.7,
            alpha=1.0 if is_ours else 0.9,
        )

    ax.set(title=title, xlabel="Epoch", ylabel=ylabel, xlim=(1, EPOCHS))
    ax.set_xticks(range(0, EPOCHS + 1, 10))
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    data = {name: read_results(path) for name, path in RESULT_FILES.items()}
    data["MSA-YOLO (Ours)"] = read_msa_results(MSA_STAGE1, MSA_STAGE2)

    for metric in METRICS:
        print(f"已保存：{plot_metric(data, metric)}")


if __name__ == "__main__":
    main()
