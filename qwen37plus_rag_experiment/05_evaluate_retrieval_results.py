# -*- coding: utf-8 -*-
"""评价RAG检索召回与模型候选选择结果。

该脚本与04_evaluate_rag_results.py互补：
- 04脚本使用与直接基线相同的主识别指标；
- 05脚本专门计算知识库Top-1/Top-3/Top-5召回率、MRR、SKU选择覆盖率、
  模型选择准确率和检索后重排准确率。

评价阶段允许读取人工真值；推理阶段脚本03不会读取人工真值。

运行：
    python 05_evaluate_retrieval_results.py

依赖：
    pip install pandas openpyxl pyyaml

兼容：Python 3.8+
"""

from __future__ import print_function

import argparse
import json
import math
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
    import yaml
except ImportError as exc:
    raise SystemExit(
        "缺少依赖。请先执行：pip install pandas openpyxl pyyaml\n"
        "原始错误：{}".format(exc)
    )


OUTPUT_ROOT = Path(r"E:\EdgeDownload\ultralytics-main\qwen37plus_rag_experiment")
RUN_NAME = "qwen37plus_rag_v1"
DEFAULT_CONFIG_PATH = OUTPUT_ROOT / "runs" / RUN_NAME / "config.yaml"

TRUE_VALUES = {"1", "true", "yes", "y", "是", "有"}
FALSE_VALUES = {"0", "false", "no", "n", "否", "无", ""}
CONFIRMED_VALUES = {"confirmed", "confirm", "已确认", "确定", "有效"}


# -----------------------------------------------------------------------------
# 通用辅助函数
# -----------------------------------------------------------------------------
def now_iso() -> str:
    """返回带本地时区的ISO时间字符串。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_yaml(path: Path) -> Dict[str, Any]:
    """读取YAML配置文件。"""
    if not path.exists():
        raise FileNotFoundError("未找到配置文件：{}".format(path))
    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError("config.yaml内容无效")
    return config


def clean_cell(value: Any) -> str:
    """将表格值转换为稳定字符串。"""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return "{:.0f}".format(value)
        return format(value, "g")
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "null"}:
        return ""
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def normalize_sku(value: Any) -> str:
    """标准化SKU编号，消除空格、标点和大小写差异。"""
    text = clean_cell(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def parse_bool(value: Any) -> Optional[bool]:
    """解析真值表中的0/1、是/否和布尔文本。"""
    text = clean_cell(value).casefold()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def parse_candidates(value: Any) -> List[Dict[str, Any]]:
    """解析预测表中的retrieved_candidates_json字段。"""
    text = clean_cell(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def metric_record(numerator: int, denominator: int) -> Dict[str, Any]:
    """构造包含分子、分母和比例值的统一指标记录。"""
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": None if denominator == 0 else float(numerator) / float(denominator),
    }


def safe_mean(values: List[float]) -> Optional[float]:
    """计算非空列表均值。"""
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def first_existing_column(frame: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """从候选字段名中返回数据表实际存在的第一个字段。"""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def resolve_gt_columns(
    gt_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    """根据配置解析真值表字段名。"""
    configured = config.get("evaluation", {}).get("column_candidates", {})
    defaults = {
        "gt_sku_id": ["gt_sku_id"],
        "label_status": ["label_status"],
        "readable": ["readable"],
        "visibility": ["visibility"],
        "multi_object": ["multi_object"],
        "false_positive": ["false_positive"],
        "gt_drug_name": ["gt_drug_name"],
    }
    resolved: Dict[str, Optional[str]] = {}
    for logical_name, fallback in defaults.items():
        candidates = configured.get(logical_name, fallback)
        resolved[logical_name] = first_existing_column(gt_df, list(candidates))
    if not resolved["gt_sku_id"]:
        raise KeyError("真值表中未找到gt_sku_id字段")
    return resolved


def attach_ground_truth(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> pd.DataFrame:
    """按照输入清单冻结的gt_row_index将预测与人工真值合并。"""
    required_manifest = {"sample_id", "gt_match_status", "gt_row_index"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise KeyError("输入清单缺少字段：{}".format(sorted(missing_manifest)))
    if "sample_id" not in predictions.columns:
        raise KeyError("预测结果缺少sample_id字段")

    if manifest["sample_id"].duplicated().any():
        raise ValueError("输入清单存在重复sample_id")
    if predictions["sample_id"].duplicated().any():
        raise ValueError("预测结果存在重复sample_id")

    merged = manifest.merge(
        predictions,
        on="sample_id",
        how="left",
        validate="one_to_one",
        suffixes=("_manifest", ""),
    )

    gt_rows: List[Dict[str, Any]] = []
    for record in merged.to_dict(orient="records"):
        gt_record: Dict[str, Any] = {}
        match_status = clean_cell(record.get("gt_match_status"))
        row_index_text = clean_cell(record.get("gt_row_index"))
        if match_status == "matched" and row_index_text:
            row_index = int(float(row_index_text))
            if row_index < 0 or row_index >= len(gt_df):
                raise IndexError(
                    "sample_id={}的gt_row_index越界：{}".format(
                        record.get("sample_id"), row_index
                    )
                )
            gt_series = gt_df.iloc[row_index]
            gt_record = {
                "gt__{}".format(column): gt_series.get(column)
                for column in gt_df.columns
            }
        gt_rows.append(gt_record)

    gt_attached = pd.DataFrame(gt_rows)
    return pd.concat([merged.reset_index(drop=True), gt_attached], axis=1)


def candidate_skus(candidates: List[Dict[str, Any]]) -> List[str]:
    """按候选排名提取标准化SKU列表。"""
    ordered = sorted(
        candidates,
        key=lambda item: int(item.get("rank", 10 ** 9)),
    )
    return [
        normalize_sku(item.get("sku_id"))
        for item in ordered
        if normalize_sku(item.get("sku_id"))
    ]


def retrieval_rank(gt_sku: str, candidates: List[Dict[str, Any]]) -> Optional[int]:
    """返回真值SKU在候选列表中的1起始排名。"""
    for position, candidate_sku in enumerate(candidate_skus(candidates), start=1):
        if candidate_sku == gt_sku:
            return position
    return None


def build_subset_metrics(frame: pd.DataFrame, top_k: int) -> Dict[str, Any]:
    """计算指定子集的检索与SKU选择指标。"""
    count = int(len(frame))
    top1 = int(frame["retrieval_hit_at_1"].sum()) if count else 0
    top3 = int(frame["retrieval_hit_at_3"].sum()) if count else 0
    top5 = int(frame["retrieval_hit_at_5"].sum()) if count else 0
    topk = int(frame["retrieval_hit_at_k"].sum()) if count else 0
    selected_count = int(frame["sku_selected"].sum()) if count else 0
    selected_correct = int(frame["selected_sku_correct"].sum()) if count else 0
    retrieved_count = int(frame["retrieval_hit_at_k"].sum()) if count else 0
    rerank_correct = int(
        (frame["retrieval_hit_at_k"] & frame["selected_sku_correct"]).sum()
    ) if count else 0

    reciprocal_ranks = [
        1.0 / float(rank)
        for rank in frame["gt_candidate_rank"].dropna().astype(int).tolist()
        if rank > 0
    ]

    return {
        "sample_count": count,
        "retrieval_recall_at_1": metric_record(top1, count),
        "retrieval_recall_at_3": metric_record(top3, count),
        "retrieval_recall_at_5": metric_record(top5, count),
        "retrieval_recall_at_k": {
            "k": int(top_k),
            **metric_record(topk, count),
        },
        "mean_reciprocal_rank": safe_mean(reciprocal_ranks),
        "sku_selection_coverage": metric_record(selected_count, count),
        "selected_sku_accuracy_all": metric_record(selected_correct, count),
        "selected_sku_accuracy_when_selected": metric_record(
            selected_correct, selected_count
        ),
        "reranker_accuracy_given_gt_retrieved": metric_record(
            rerank_correct, retrieved_count
        ),
    }


def flatten_metrics(metrics: Dict[str, Any]) -> pd.DataFrame:
    """将嵌套指标展开为便于论文整理的CSV。"""
    rows: List[Dict[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            if {"numerator", "denominator", "value"}.issubset(value.keys()):
                rows.append(
                    {
                        "metric": prefix,
                        "value": value.get("value"),
                        "numerator": value.get("numerator"),
                        "denominator": value.get("denominator"),
                    }
                )
                return
            for key, child in value.items():
                child_prefix = "{}.{}".format(prefix, key) if prefix else str(key)
                visit(child_prefix, child)
        elif isinstance(value, (int, float, str)) or value is None:
            rows.append(
                {
                    "metric": prefix,
                    "value": value,
                    "numerator": "",
                    "denominator": "",
                }
            )

    visit("", metrics)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 主评价流程
# -----------------------------------------------------------------------------
def main() -> int:
    """合并预测与真值，计算检索和模型选SKU指标。"""
    parser = argparse.ArgumentParser(description="评价RAG检索与候选选择结果")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = read_yaml(Path(args.config))
    paths = config["paths"]
    top_k = int(config.get("retrieval", {}).get("top_k", 5))

    manifest_path = Path(paths["manifest_path"])
    predictions_path = Path(paths["predictions_path"])
    gt_path = Path(paths["ground_truth_path"])
    merged_path = Path(paths["retrieval_merged_results_path"])
    metrics_path = Path(paths["retrieval_metrics_path"])
    metrics_csv_path = Path(paths["retrieval_metrics_csv_path"])
    error_cases_path = Path(paths["retrieval_error_cases_path"])
    gt_sheet = config.get("data", {}).get("ground_truth_sheet", "instances_gt")

    for label, path in [
        ("输入清单", manifest_path),
        ("RAG预测结果", predictions_path),
        ("人工真值表", gt_path),
    ]:
        if not path.exists():
            raise FileNotFoundError("{}不存在：{}".format(label, path))

    manifest = pd.read_csv(
        manifest_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    predictions = pd.read_csv(
        predictions_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    gt_df = pd.read_excel(gt_path, sheet_name=gt_sheet, engine="openpyxl")
    gt_cols = resolve_gt_columns(gt_df, config)

    merged = attach_ground_truth(manifest, predictions, gt_df)

    gt_sku_column = "gt__{}".format(gt_cols["gt_sku_id"])
    label_status_column = (
        "gt__{}".format(gt_cols["label_status"])
        if gt_cols.get("label_status")
        else None
    )
    false_positive_column = (
        "gt__{}".format(gt_cols["false_positive"])
        if gt_cols.get("false_positive")
        else None
    )
    multi_object_column = (
        "gt__{}".format(gt_cols["multi_object"])
        if gt_cols.get("multi_object")
        else None
    )
    readable_column = (
        "gt__{}".format(gt_cols["readable"])
        if gt_cols.get("readable")
        else None
    )
    visibility_column = (
        "gt__{}".format(gt_cols["visibility"])
        if gt_cols.get("visibility")
        else None
    )

    merged["gt_sku_normalized"] = merged[gt_sku_column].apply(normalize_sku)
    merged["pred_sku_normalized"] = merged.get(
        "pred_sku_id", pd.Series([""] * len(merged))
    ).apply(normalize_sku)
    merged["retrieved_candidates"] = merged.get(
        "retrieved_candidates_json", pd.Series([""] * len(merged))
    ).apply(parse_candidates)
    merged["gt_candidate_rank"] = merged.apply(
        lambda row: retrieval_rank(
            row["gt_sku_normalized"], row["retrieved_candidates"]
        )
        if row["gt_sku_normalized"]
        else None,
        axis=1,
    )

    merged["retrieval_hit_at_1"] = merged["gt_candidate_rank"].eq(1)
    merged["retrieval_hit_at_3"] = merged["gt_candidate_rank"].apply(
        lambda value: bool(pd.notna(value) and int(value) <= 3)
    )
    merged["retrieval_hit_at_5"] = merged["gt_candidate_rank"].apply(
        lambda value: bool(pd.notna(value) and int(value) <= 5)
    )
    merged["retrieval_hit_at_k"] = merged["gt_candidate_rank"].apply(
        lambda value: bool(pd.notna(value) and int(value) <= top_k)
    )
    merged["sku_selected"] = merged["pred_sku_normalized"].astype(str).ne("")
    merged["selected_sku_correct"] = (
        merged["sku_selected"]
        & merged["gt_sku_normalized"].astype(str).ne("")
        & merged["pred_sku_normalized"].eq(merged["gt_sku_normalized"])
    )

    eligible = merged["gt_sku_normalized"].astype(str).ne("")
    eligible &= merged["gt_match_status"].astype(str).eq("matched")
    if label_status_column:
        eligible &= merged[label_status_column].apply(
            lambda value: clean_cell(value).casefold() in CONFIRMED_VALUES
        )
    if false_positive_column:
        eligible &= merged[false_positive_column].apply(parse_bool).ne(True)

    merged["retrieval_evaluable"] = eligible
    if multi_object_column:
        merged["is_single_object"] = merged[multi_object_column].apply(parse_bool).ne(True)
    else:
        merged["is_single_object"] = True

    evaluable = merged.loc[merged["retrieval_evaluable"]].copy()
    single_object = evaluable.loc[evaluable["is_single_object"]].copy()

    metrics: Dict[str, Any] = {
        "experiment": {
            "name": config["experiment"]["name"],
            "evaluated_at": now_iso(),
            "model": config["api"]["model"],
            "retrieval_method": config["retrieval"]["method"],
            "top_k": top_k,
            "query_source": config["retrieval"].get("query_source"),
            "ground_truth_used_during_inference": False,
            "evaluation_script_version": "1.0",
        },
        "sample_counts": {
            "manifest_total": int(len(merged)),
            "prediction_ok": int((merged.get("status", "") == "ok").sum()),
            "retrieval_evaluable_with_gt_sku": int(len(evaluable)),
            "single_object_retrieval_evaluable": int(len(single_object)),
            "excluded_without_gt_sku_or_confirmation": int(len(merged) - len(evaluable)),
        },
        "all_gt_sku_samples": build_subset_metrics(evaluable, top_k),
        "single_object_gt_sku_samples": build_subset_metrics(single_object, top_k),
        "breakdown_by_readable": {},
        "breakdown_by_visibility": {},
    }

    if readable_column:
        for value, subset in evaluable.groupby(
            evaluable[readable_column].apply(clean_cell).replace("", "missing")
        ):
            metrics["breakdown_by_readable"][str(value)] = build_subset_metrics(
                subset, top_k
            )

    if visibility_column:
        for value, subset in evaluable.groupby(
            evaluable[visibility_column].apply(clean_cell).replace("", "missing")
        ):
            metrics["breakdown_by_visibility"][str(value)] = build_subset_metrics(
                subset, top_k
            )

    def classify_error(row: pd.Series) -> str:
        """按照检索失败和候选选择失败划分错误类型。"""
        if not bool(row["retrieval_evaluable"]):
            return "not_evaluable"
        if not bool(row["retrieval_hit_at_k"]):
            return "retrieval_miss"
        if not bool(row["sku_selected"]):
            return "gt_retrieved_but_no_sku_selected"
        if not bool(row["selected_sku_correct"]):
            return "gt_retrieved_but_wrong_sku_selected"
        return "correct"

    merged["retrieval_error_type"] = merged.apply(classify_error, axis=1)

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    export_merged = merged.drop(columns=["retrieved_candidates"], errors="ignore")
    export_merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    export_merged.loc[
        export_merged["retrieval_error_type"].isin(
            {
                "retrieval_miss",
                "gt_retrieved_but_no_sku_selected",
                "gt_retrieved_but_wrong_sku_selected",
            }
        )
    ].to_csv(error_cases_path, index=False, encoding="utf-8-sig")

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    flatten_metrics(metrics).to_csv(
        metrics_csv_path, index=False, encoding="utf-8-sig"
    )

    overall = metrics["all_gt_sku_samples"]
    print("评价完成。")
    print("可评价GT-SKU样本：{}".format(overall["sample_count"]))
    for label, key in [
        ("Top-1召回率", "retrieval_recall_at_1"),
        ("Top-3召回率", "retrieval_recall_at_3"),
        ("Top-5召回率", "retrieval_recall_at_5"),
        ("最终SKU选择准确率", "selected_sku_accuracy_all"),
        ("已选择样本中的SKU准确率", "selected_sku_accuracy_when_selected"),
        ("真值已召回条件下的重排准确率", "reranker_accuracy_given_gt_retrieved"),
    ]:
        item = overall[key]
        display = "N/A" if item["value"] is None else "{:.2%}".format(item["value"])
        print("{}：{}（{}/{}）".format(
            label, display, item["numerator"], item["denominator"]
        ))
    mrr = overall["mean_reciprocal_rank"]
    print("MRR：{}".format("N/A" if mrr is None else "{:.4f}".format(mrr)))
    print("合并结果：{}".format(merged_path))
    print("检索指标：{}".format(metrics_path))
    print("错误样本：{}".format(error_cases_path))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("\n检索评价失败：{}".format(error), file=sys.stderr)
        sys.exit(1)
