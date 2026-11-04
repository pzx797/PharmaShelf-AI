"""Plot Recall@K from the RAG retrieval evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_METRICS = (
    PROJECT_ROOT / "qwen37plus_rag_experiment" / "runs" / "qwen37plus_rag_v1" / "evaluation" / "retrieval_metrics.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "topk_recall_plot"


def configure_fonts() -> None:
    """Choose an installed Chinese font and keep minus signs readable."""
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_recall_values(metrics_path: Path) -> tuple[list[int], list[float]]:
    """Read Recall@1, Recall@3 and Recall@5 as percentages."""
    if not metrics_path.is_file():
        raise FileNotFoundError(f"找不到检索评估文件：{metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    section = metrics.get("all_gt_sku_samples")
    if not isinstance(section, dict):
        raise KeyError("指标文件缺少 all_gt_sku_samples 字段")

    top_k = [1, 3, 5]
    recalls = []
    for k in top_k:
        key = f"retrieval_recall_at_{k}"
        record = section.get(key)
        if not isinstance(record, dict) or record.get("value") is None:
            raise KeyError(f"指标文件缺少 {key}.value")
        recalls.append(float(record["value"]) * 100.0)
    return top_k, recalls


def plot_recall(top_k: list[int], recalls: list[float], output_dir: Path, metrics_path: Path) -> None:
    """Create publication-ready PNG, PDF and SVG figures."""
    configure_fonts()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)

    ax.plot(
        top_k,
        recalls,
        color="#2878B5",
        marker="o",
        markersize=7,
        linewidth=2.2,
        markerfacecolor="white",
        markeredgewidth=2.0,
    )

    for k, recall in zip(top_k, recalls):
        ax.annotate(
            f"{recall:.2f}%",
            xy=(k, recall),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_xlabel("候选数量 K", fontsize=12)
    ax.set_ylabel("Recall@K（%）", fontsize=12)
    ax.set_xticks(top_k)
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "topk_sku_recall"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    print("读取指标：", metrics_path.resolve())
    print(f"绘图数据：{dict(zip(top_k, [round(v, 2) for v in recalls]))}")
    for suffix in (".png", ".pdf", ".svg"):
        print("已保存：", stem.with_suffix(suffix))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制不同Top-K下的SKU检索召回率")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help="retrieval_metrics.json 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="图片输出目录",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ks, values = load_recall_values(args.metrics)
    plot_recall(ks, values, args.output_dir, args.metrics)
