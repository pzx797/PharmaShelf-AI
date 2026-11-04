# -*- coding: utf-8 -*-
"""Evaluate Qwen3.7-Plus direct recognition results against the enriched GT table.

This script does not call the model. It:
- Merges predictions with the frozen ground-truth workbook.
- Calculates core/full drug-name, specification, manufacturer, strict-SKU, refusal,
  unsupported-definitive-answer, format-validity, API-success, latency, and
  token metrics.
- Implements strict SKU scheme B: the model never sees the knowledge base;
  the evaluator maps predicted name + specification + manufacturer to a unique
  knowledge-base SKU after inference.
- Produces merged_results.csv, metrics.json, and error_cases.csv.

Run:
    python 03_evaluate_direct_results.py

Dependencies:
    pip install pandas openpyxl pyyaml

Python compatibility: Python 3.8+
"""

from __future__ import print_function

import argparse
import ast
import json
import math
import re
import statistics
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import pandas as pd
    import yaml
except ImportError as exc:
    raise SystemExit(
        "缺少依赖。请先执行：pip install pandas openpyxl pyyaml\n"
        "原始错误：{}".format(exc)
    )


OUTPUT_ROOT = Path(
    r"E:\EdgeDownload\ultralytics-main\qwen37plus_direct_experiment"
)
RUN_NAME = "qwen37plus_direct_v1"
DEFAULT_CONFIG_PATH = OUTPUT_ROOT / "runs" / RUN_NAME / "config.yaml"

TRUE_VALUES = {"1", "true", "yes", "y", "是", "有", "正确"}
FALSE_VALUES = {"0", "false", "no", "n", "否", "无", ""}
UNKNOWN_LABEL_VALUES = {
    "unknown",
    "unconfirmed",
    "uncertain",
    "无法确认",
    "未知",
    "待确认",
    "不确定",
}
CONFIRMED_LABEL_VALUES = {"confirmed", "confirm", "已确认", "确定", "有效"}


# 仅用于“核心药品名称”评价的剂型/包装后缀。
# 规则是保守的：只允许去除完整名称末尾的已知后缀，不做任意子串匹配。
DOSAGE_FORM_SUFFIXES = tuple(
    sorted(
        {
            "缓释胶囊", "肠溶胶囊", "软胶囊", "硬胶囊", "胶囊剂", "胶囊",
            "缓释片", "控释片", "肠溶片", "分散片", "咀嚼片", "泡腾片",
            "口崩片", "舌下片", "含片", "薄膜衣片", "片剂", "片",
            "颗粒剂", "颗粒", "冲剂",
            "浓缩丸", "水蜜丸", "大蜜丸", "小蜜丸", "水丸", "蜜丸",
            "滴丸", "丸剂", "丸",
            "口服溶液", "口服液", "合剂", "糖浆剂", "糖浆",
            "混悬液", "混悬剂", "乳剂", "溶液剂", "溶液",
            "喷雾剂", "喷剂", "气雾剂",
            "软膏剂", "软膏", "乳膏剂", "乳膏", "凝胶剂", "凝胶",
            "贴膏剂", "贴膏", "贴剂", "膏药",
            "散剂", "散", "粉剂", "粉",
            "滴眼液", "滴耳液", "滴鼻液",
            "注射液", "注射剂",
            "栓剂", "栓", "酊剂", "酊", "洗剂", "搽剂", "膜剂",
            "茶剂", "饮片",
        },
        key=len,
        reverse=True,
    )
)

# 前10个样本曾用于提示词调试。脚本会同时报告全量结果和排除开发样本后的盲测结果。
DEFAULT_PROMPT_DEVELOPMENT_SAMPLE_COUNT = 10


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            "未找到配置文件：{}。请先运行01_prepare_direct_experiment.py".format(path)
        )
    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError("config.yaml内容无效")
    return config


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "null"}:
        return ""
    # Excel often renders integer identifiers as 123.0.
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def normalize_basic(value: Any) -> str:
    text = clean_cell(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\s\u3000]+", "", text)
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"[·•,，。.;；:：'\"“”‘’/\\_-]+", "", text)
    return text


def normalize_drug_name(value: Any) -> str:
    text = normalize_basic(value)
    text = re.sub(r"\(otc\)", "", text)
    return text


def normalize_specification(value: Any) -> str:
    text = clean_cell(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).casefold()
    replacements = {
        "毫克": "mg",
        "克": "g",
        "毫升": "ml",
        "升": "l",
        "微克": "μg",
        "片剂": "片",
        "胶囊剂": "粒",
        "袋装": "袋",
        "支装": "支",
        "×": "x",
        "✕": "x",
        "＊": "x",
        "*": "x",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"[\s\u3000]+", "", text)
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"[，,；;。]+", "", text)
    text = re.sub(r"\.0(?=[a-zμ]|x|片|粒|袋|支|瓶|盒|包|贴)", "", text)
    return text


def normalize_manufacturer(value: Any) -> str:
    text = normalize_basic(value)
    if not text:
        return ""
    # Only standardize clearly equivalent legal suffixes; do not remove them all.
    text = text.replace("有限责任公司", "有限公司")
    text = text.replace("股份有限责任公司", "股份有限公司")
    return text


def parse_bool(value: Any) -> Optional[bool]:
    text = clean_cell(value).casefold()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    try:
        number = float(text)
        return number != 0
    except (TypeError, ValueError):
        return None


def first_existing_column(
    frame: pd.DataFrame, candidates: Sequence[str], required: bool = False
) -> Optional[str]:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise KeyError(
            "未找到字段{}。当前表头：{}".format(
                list(candidates), "、".join(map(str, frame.columns.tolist()))
            )
        )
    return None


def resolve_columns(
    frame: pd.DataFrame, candidate_map: Dict[str, Sequence[str]]
) -> Dict[str, Optional[str]]:
    return {
        logical_name: first_existing_column(frame, candidates, required=False)
        for logical_name, candidates in candidate_map.items()
    }


def split_aliases(value: Any) -> List[str]:
    text = clean_cell(value)
    if not text:
        return []

    # Prefer JSON/Python list syntax when available.
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [clean_cell(item) for item in parsed if clean_cell(item)]
            except Exception:
                pass

    parts = re.split(r"[|｜;；,，、/\n]+", text)
    return [part.strip() for part in parts if part.strip()]


def normalize_sku(value: Any) -> str:
    return normalize_basic(value)


def build_kb_records(
    kb_df: pd.DataFrame, column_map: Dict[str, Sequence[str]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    sku_col = first_existing_column(kb_df, column_map["sku_id"], required=True)
    name_cols = [column for column in column_map["drug_name"] if column in kb_df.columns]
    spec_cols = [
        column for column in column_map["specification"] if column in kb_df.columns
    ]
    manufacturer_cols = [
        column for column in column_map["manufacturer"] if column in kb_df.columns
    ]
    alias_cols = [column for column in column_map["aliases"] if column in kb_df.columns]

    if not name_cols:
        raise KeyError("知识库中没有可用的药品名称字段")

    records: List[Dict[str, Any]] = []
    by_sku: Dict[str, Dict[str, Any]] = {}

    for index, row in kb_df.iterrows():
        raw_sku = clean_cell(row.get(sku_col))
        sku_norm = normalize_sku(raw_sku)
        if not sku_norm:
            continue

        names: Set[str] = set()
        raw_names: Set[str] = set()
        for column in name_cols:
            raw = clean_cell(row.get(column))
            if raw:
                raw_names.add(raw)
                normalized = normalize_drug_name(raw)
                if normalized:
                    names.add(normalized)
        for column in alias_cols:
            for alias in split_aliases(row.get(column)):
                raw_names.add(alias)
                normalized = normalize_drug_name(alias)
                if normalized:
                    names.add(normalized)

        specs: Set[str] = set()
        raw_specs: Set[str] = set()
        for column in spec_cols:
            raw = clean_cell(row.get(column))
            if raw:
                raw_specs.add(raw)
                normalized = normalize_specification(raw)
                if normalized:
                    specs.add(normalized)

        manufacturers: Set[str] = set()
        raw_manufacturers: Set[str] = set()
        for column in manufacturer_cols:
            raw = clean_cell(row.get(column))
            if raw:
                raw_manufacturers.add(raw)
                normalized = normalize_manufacturer(raw)
                if normalized:
                    manufacturers.add(normalized)

        record = {
            "kb_row_index": int(index),
            "sku_id": raw_sku,
            "sku_norm": sku_norm,
            "names": names,
            "raw_names": sorted(raw_names),
            "specs": specs,
            "raw_specs": sorted(raw_specs),
            "manufacturers": manufacturers,
            "raw_manufacturers": sorted(raw_manufacturers),
        }
        records.append(record)
        # Keep the first row for a duplicated normalized SKU; duplicates remain in records.
        by_sku.setdefault(sku_norm, record)

    return records, by_sku


def remove_parenthetical_qualifiers(normalized_name: str) -> str:
    """去除括号及括号内的品牌、品种或补充限定信息。

    仅用于核心名称评价，例如：
    菊花(贡菊) -> 菊花
    复方氨酚烷胺胶囊(快克) -> 复方氨酚烷胺胶囊
    """
    if not normalized_name:
        return ""
    previous = None
    current = normalized_name
    while previous != current:
        previous = current
        current = re.sub(r"\([^()]*\)", "", current)
    return current


def strip_terminal_dosage_form(normalized_name: str) -> str:
    """保守地去除末尾剂型/包装后缀，不做任意前缀或子串匹配。"""
    if not normalized_name:
        return ""

    for suffix in DOSAGE_FORM_SUFFIXES:
        if normalized_name.endswith(suffix):
            base = normalized_name[: -len(suffix)]
            # 至少保留2个字符，避免将“片”等极短名称处理为空或过短。
            if len(base) >= 2:
                return base
    return normalized_name


def drug_name_core_variants(value: Any) -> Set[str]:
    """生成用于核心药品名称评价的保守等价形式。

    允许：
    - 去除括号补充信息：菊花(贡菊) -> 菊花
    - 去除末尾剂型：乳癖消片 -> 乳癖消

    不允许：
    - 任意前缀/包含匹配
    - 编辑距离或模糊字符串相似度
    - 删除名称中间的有效成分
    """
    full_name = normalize_drug_name(value)
    if not full_name:
        return set()

    variants: Set[str] = {full_name}
    without_parentheses = remove_parenthetical_qualifiers(full_name)
    if without_parentheses:
        variants.add(without_parentheses)

    # 只对去括号后的名称移除一个末尾剂型后缀。
    dosage_core = strip_terminal_dosage_form(without_parentheses)
    if dosage_core:
        variants.add(dosage_core)

    return {item for item in variants if item}


def prediction_full_name_set(row: pd.Series) -> Set[str]:
    values = [row.get("pred_drug_name", ""), row.get("pred_generic_name", "")]
    return {
        normalize_drug_name(value)
        for value in values
        if normalize_drug_name(value)
    }


def prediction_core_name_set(row: pd.Series) -> Set[str]:
    names: Set[str] = set()
    for value in [row.get("pred_drug_name", ""), row.get("pred_generic_name", "")]:
        names.update(drug_name_core_variants(value))
    return names


def gt_full_name_set(
    gt_row: pd.Series,
    gt_drug_name_col: Optional[str],
    gt_sku_id_col: Optional[str],
    kb_by_sku: Dict[str, Dict[str, Any]],
) -> Set[str]:
    names: Set[str] = set()

    if gt_drug_name_col:
        normalized = normalize_drug_name(gt_row.get(gt_drug_name_col))
        if normalized:
            names.add(normalized)

    if gt_sku_id_col:
        sku_norm = normalize_sku(gt_row.get(gt_sku_id_col))
        record = kb_by_sku.get(sku_norm)
        if record:
            names.update(record["names"])

    return names


def gt_core_name_set(
    gt_row: pd.Series,
    gt_drug_name_col: Optional[str],
    gt_sku_id_col: Optional[str],
    kb_by_sku: Dict[str, Dict[str, Any]],
) -> Set[str]:
    names: Set[str] = set()

    if gt_drug_name_col:
        names.update(drug_name_core_variants(gt_row.get(gt_drug_name_col)))

    if gt_sku_id_col:
        sku_norm = normalize_sku(gt_row.get(gt_sku_id_col))
        record = kb_by_sku.get(sku_norm)
        if record:
            for raw_name in record.get("raw_names", []):
                names.update(drug_name_core_variants(raw_name))

    return names


# 以下两个函数保留给严格SKU离线映射使用。
# 严格SKU映射不采用“去剂型后的核心名称”，只使用完整标准名/合法别名。
def prediction_name_set(row: pd.Series) -> Set[str]:
    return prediction_full_name_set(row)


def gt_valid_names(
    gt_row: pd.Series,
    gt_drug_name_col: Optional[str],
    gt_sku_id_col: Optional[str],
    kb_by_sku: Dict[str, Dict[str, Any]],
) -> Set[str]:
    return gt_full_name_set(
        gt_row,
        gt_drug_name_col,
        gt_sku_id_col,
        kb_by_sku,
    )


def map_prediction_to_unique_sku(
    row: pd.Series, kb_records: List[Dict[str, Any]]
) -> Tuple[str, str, int]:
    """Map predicted identity fields to a unique KB SKU using exact normalized fields.

    Returns: (mapped_sku_id, mapping_status, candidate_count)
    """
    pred_names = prediction_name_set(row)
    pred_spec = normalize_specification(row.get("pred_specification", ""))
    pred_manufacturer = normalize_manufacturer(row.get("pred_manufacturer", ""))

    if not pred_names:
        return "", "no_predicted_name", 0

    candidates = [record for record in kb_records if pred_names & record["names"]]
    if not candidates:
        return "", "name_not_found", 0

    if pred_spec:
        candidates = [record for record in candidates if pred_spec in record["specs"]]
        if not candidates:
            return "", "specification_not_found", 0

    if pred_manufacturer:
        candidates = [
            record
            for record in candidates
            if pred_manufacturer in record["manufacturers"]
        ]
        if not candidates:
            return "", "manufacturer_not_found", 0

    unique_skus = {}
    for record in candidates:
        unique_skus[record["sku_norm"]] = record["sku_id"]

    if len(unique_skus) == 1:
        return list(unique_skus.values())[0], "unique", 1
    return "", "ambiguous", len(unique_skus)


def expected_decision(
    gt_row: pd.Series,
    gt_cols: Dict[str, Optional[str]],
) -> str:
    false_positive = (
        parse_bool(gt_row.get(gt_cols["false_positive"]))
        if gt_cols.get("false_positive")
        else False
    )
    multi_object = (
        parse_bool(gt_row.get(gt_cols["multi_object"]))
        if gt_cols.get("multi_object")
        else False
    )
    label_status = (
        clean_cell(gt_row.get(gt_cols["label_status"])).casefold()
        if gt_cols.get("label_status")
        else ""
    )
    gt_name = (
        clean_cell(gt_row.get(gt_cols["gt_drug_name"]))
        if gt_cols.get("gt_drug_name")
        else ""
    )

    if false_positive is True:
        return "non_drug"
    if multi_object is True:
        return "multi_object"
    if label_status in UNKNOWN_LABEL_VALUES or not gt_name:
        return "unknown"
    return "matched"


def is_main_identification_sample(
    gt_row: pd.Series,
    gt_cols: Dict[str, Optional[str]],
) -> bool:
    if expected_decision(gt_row, gt_cols) != "matched":
        return False
    label_status = (
        clean_cell(gt_row.get(gt_cols["label_status"])).casefold()
        if gt_cols.get("label_status")
        else ""
    )
    if label_status and label_status in UNKNOWN_LABEL_VALUES:
        return False
    return True


def proportion(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def metric_record(numerator: int, denominator: int) -> Dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": proportion(numerator, denominator),
    }


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def safe_mean(series: pd.Series) -> Optional[float]:
    return None if series.empty else float(series.mean())


def safe_sum(series: pd.Series) -> Optional[float]:
    return None if series.empty else float(series.sum())


def classify_subset(row: pd.Series, gt_cols: Dict[str, Optional[str]]) -> str:
    exp = row.get("expected_decision", "")
    if exp == "non_drug":
        return "false_positive"
    if exp == "multi_object":
        return "multi_object"
    if exp == "unknown":
        return "unknown"

    readable = (
        parse_bool(row.get(gt_cols["readable"])) if gt_cols.get("readable") else None
    )
    visibility = (
        clean_cell(row.get(gt_cols["visibility"])).casefold()
        if gt_cols.get("visibility")
        else ""
    )
    if readable is False or visibility in {
        "partial",
        "low",
        "poor",
        "部分可见",
        "低",
        "较差",
        "遮挡",
    }:
        return "difficult"
    return "clear_or_regular"


def determine_error_types(row: pd.Series, gt_cols: Dict[str, Optional[str]]) -> List[str]:
    errors: List[str] = []
    status = clean_cell(row.get("status"))
    decision = clean_cell(row.get("decision"))
    expected = clean_cell(row.get("expected_decision"))

    if status == "format_error":
        errors.append("E11_JSON格式错误")
    elif status in {"api_error", "input_error"}:
        errors.append("E12_API或输入错误")

    if expected == "non_drug" and decision not in {"non_drug", "api_error", "format_error"}:
        errors.append("E10_YOLO误检框未被拒识")
    if expected == "multi_object" and decision not in {
        "multi_object",
        "api_error",
        "format_error",
    }:
        errors.append("E9_多目标图像处理错误")
    if expected == "unknown" and decision == "matched":
        errors.append("E8_不可确认样本被强行识别")
    if expected == "matched" and decision in {"unknown", "non_drug", "multi_object"}:
        errors.append("E7_已确认药品被拒识或决策类型错误")

    if expected == "matched" and decision == "matched":
        if row.get("name_core_correct") is False:
            errors.append("E3_核心药品名称错误或相似包装混淆")
        elif (
            row.get("name_core_correct") is True
            and row.get("name_full_correct") is False
        ):
            errors.append("E3B_核心名称正确但完整商品名不一致")
        if row.get("name_core_correct") is True and row.get("spec_correct") is False:
            errors.append("E4_核心药名正确但规格错误或缺失")
        if (
            row.get("name_core_correct") is True
            and row.get("manufacturer_correct") is False
        ):
            errors.append("E5_核心药名正确但厂家错误或缺失")
        if row.get("strict_sku_correct") is False:
            errors.append("E6_严格SKU无法唯一匹配或错误")

    readable = (
        parse_bool(row.get(gt_cols["readable"])) if gt_cols.get("readable") else None
    )
    visibility = (
        clean_cell(row.get(gt_cols["visibility"])).casefold()
        if gt_cols.get("visibility")
        else ""
    )
    if errors and readable is False:
        errors.insert(0, "E1_文字模糊或不可读")
    if errors and visibility in {"partial", "部分可见", "遮挡", "occluded"}:
        errors.insert(0, "E2_遮挡或裁剪不完整")

    unique: List[str] = []
    for item in errors:
        if item not in unique:
            unique.append(item)
    return unique


def sample_number(sample_id: Any) -> Optional[int]:
    match = re.fullmatch(r"sample_(\d+)", clean_cell(sample_id))
    if not match:
        return None
    return int(match.group(1))


def is_prompt_development_sample(sample_id: Any, development_count: int) -> bool:
    number = sample_number(sample_id)
    return number is not None and 1 <= number <= development_count


def dataframe_bool_metric(frame: pd.DataFrame, column: str) -> Dict[str, Any]:
    valid = frame[column].notna()
    numerator = int((frame.loc[valid, column] == True).sum())  # noqa: E712
    denominator = int(valid.sum())
    return metric_record(numerator, denominator)


def recognition_metric_bundle(frame: pd.DataFrame) -> Dict[str, Any]:
    return {
        # drug_name_accuracy保留为兼容名称，正式含义为“核心药品名称准确率”。
        "drug_name_accuracy": dataframe_bool_metric(frame, "name_core_correct"),
        "drug_name_core_accuracy": dataframe_bool_metric(
            frame, "name_core_correct"
        ),
        "drug_name_full_accuracy": dataframe_bool_metric(
            frame, "name_full_correct"
        ),
        "specification_accuracy": dataframe_bool_metric(frame, "spec_correct"),
        "manufacturer_accuracy": dataframe_bool_metric(
            frame, "manufacturer_correct"
        ),
        "strict_sku_accuracy": dataframe_bool_metric(
            frame, "strict_sku_correct"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="评价Qwen3.7-Plus直接识别实验")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config = read_yaml(Path(args.config))
    paths = config["paths"]
    eval_cfg = config["evaluation"]

    manifest_path = Path(paths["manifest_path"])
    predictions_path = Path(paths["predictions_path"])
    gt_path = Path(paths["ground_truth_path"])
    kb_path = Path(paths["knowledge_base_path"])
    merged_path = Path(paths["merged_results_path"])
    metrics_path = Path(paths["metrics_path"])
    error_cases_path = Path(paths["error_cases_path"])

    for label, path in [
        ("输入清单", manifest_path),
        ("预测结果", predictions_path),
        ("人工真值表", gt_path),
        ("知识库", kb_path),
    ]:
        if not path.exists():
            raise FileNotFoundError("{}不存在：{}".format(label, path))

    manifest = pd.read_csv(
        manifest_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    predictions = pd.read_csv(
        predictions_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    gt_df = pd.read_excel(gt_path, engine="openpyxl")
    kb_df = pd.read_excel(kb_path, engine="openpyxl")

    if predictions["sample_id"].duplicated().any():
        duplicate_ids = predictions.loc[
            predictions["sample_id"].duplicated(keep=False), "sample_id"
        ].tolist()
        raise ValueError("预测表存在重复sample_id：{}".format(duplicate_ids[:10]))

    gt_cols = resolve_columns(gt_df, eval_cfg["column_candidates"])
    if not gt_cols.get("gt_drug_name"):
        raise KeyError("真值表必须包含gt_drug_name或对应候选字段")

    kb_records, kb_by_sku = build_kb_records(
        kb_df, eval_cfg["knowledge_base_columns"]
    )

    merged = manifest.merge(predictions, on="sample_id", how="left", validate="one_to_one")

    # Attach GT rows using the frozen row index stored during preparation.
    gt_columns_renamed = {column: "gt__{}".format(column) for column in gt_df.columns}
    gt_for_merge = gt_df.rename(columns=gt_columns_renamed).copy()
    gt_for_merge["gt_row_index_numeric"] = gt_for_merge.index.astype(int)
    merged["gt_row_index_numeric"] = pd.to_numeric(
        merged["gt_row_index"], errors="coerce"
    ).astype("Int64")
    merged = merged.merge(
        gt_for_merge,
        on="gt_row_index_numeric",
        how="left",
        validate="many_to_one",
    )

    # Map logical GT names to their renamed merged columns.
    merged_gt_cols: Dict[str, Optional[str]] = {}
    for logical_name, original_col in gt_cols.items():
        merged_gt_cols[logical_name] = (
            "gt__{}".format(original_col) if original_col else None
        )

    expected_values: List[str] = []
    main_flags: List[bool] = []
    name_core_correct_values: List[Optional[bool]] = []
    name_full_correct_values: List[Optional[bool]] = []
    name_match_types: List[str] = []
    spec_correct_values: List[Optional[bool]] = []
    manufacturer_correct_values: List[Optional[bool]] = []
    mapped_skus: List[str] = []
    sku_mapping_statuses: List[str] = []
    sku_candidate_counts: List[int] = []
    strict_sku_correct_values: List[Optional[bool]] = []
    decision_correct_values: List[bool] = []

    for _, row in merged.iterrows():
        gt_present = clean_cell(row.get("gt_match_status")) == "matched"
        if not gt_present:
            expected_values.append("unavailable")
            main_flags.append(False)
            name_core_correct_values.append(None)
            name_full_correct_values.append(None)
            name_match_types.append("gt_unavailable")
            spec_correct_values.append(None)
            manufacturer_correct_values.append(None)
            mapped_skus.append("")
            sku_mapping_statuses.append("gt_unavailable")
            sku_candidate_counts.append(0)
            strict_sku_correct_values.append(None)
            decision_correct_values.append(False)
            continue

        expected = expected_decision(row, merged_gt_cols)
        expected_values.append(expected)
        main_sample = is_main_identification_sample(row, merged_gt_cols)
        main_flags.append(main_sample)

        decision = clean_cell(row.get("decision"))
        status = clean_cell(row.get("status"))
        decision_correct_values.append(status == "ok" and decision == expected)

        # 名称评价分为两层：
        # 1) 完整商品名：标准名/合法别名完整一致；
        # 2) 核心药品名：允许去除括号限定信息和末尾剂型，但不允许任意前缀匹配。
        if main_sample and clean_cell(row.get(merged_gt_cols["gt_drug_name"])):
            predicted_full_names = prediction_full_name_set(row)
            predicted_core_names = prediction_core_name_set(row)
            valid_full_names = gt_full_name_set(
                row,
                merged_gt_cols["gt_drug_name"],
                merged_gt_cols.get("gt_sku_id"),
                kb_by_sku,
            )
            valid_core_names = gt_core_name_set(
                row,
                merged_gt_cols["gt_drug_name"],
                merged_gt_cols.get("gt_sku_id"),
                kb_by_sku,
            )

            full_correct = bool(predicted_full_names & valid_full_names)
            core_correct = bool(predicted_core_names & valid_core_names)
            name_full_correct_values.append(full_correct)
            name_core_correct_values.append(core_correct)

            if full_correct:
                name_match_types.append("full")
            elif core_correct:
                name_match_types.append("core_only")
            elif not predicted_full_names:
                name_match_types.append("missing_prediction")
            else:
                name_match_types.append("wrong")
        else:
            name_core_correct_values.append(None)
            name_full_correct_values.append(None)
            name_match_types.append("not_applicable")

        gt_spec = (
            clean_cell(row.get(merged_gt_cols["gt_specification"]))
            if merged_gt_cols.get("gt_specification")
            else ""
        )
        if main_sample and gt_spec:
            spec_correct_values.append(
                normalize_specification(row.get("pred_specification", ""))
                == normalize_specification(gt_spec)
            )
        else:
            spec_correct_values.append(None)

        gt_manufacturer = (
            clean_cell(row.get(merged_gt_cols["gt_manufacturer"]))
            if merged_gt_cols.get("gt_manufacturer")
            else ""
        )
        if main_sample and gt_manufacturer:
            manufacturer_correct_values.append(
                normalize_manufacturer(row.get("pred_manufacturer", ""))
                == normalize_manufacturer(gt_manufacturer)
            )
        else:
            manufacturer_correct_values.append(None)

        mapped_sku, mapping_status, candidate_count = map_prediction_to_unique_sku(
            row, kb_records
        )
        mapped_skus.append(mapped_sku)
        sku_mapping_statuses.append(mapping_status)
        sku_candidate_counts.append(candidate_count)

        gt_sku = (
            clean_cell(row.get(merged_gt_cols["gt_sku_id"]))
            if merged_gt_cols.get("gt_sku_id")
            else ""
        )
        if main_sample and gt_sku:
            strict_sku_correct_values.append(
                bool(mapped_sku)
                and normalize_sku(mapped_sku) == normalize_sku(gt_sku)
            )
        else:
            strict_sku_correct_values.append(None)

    merged["expected_decision"] = expected_values
    merged["is_main_identification_sample"] = main_flags
    merged["decision_correct"] = decision_correct_values
    merged["name_core_correct"] = pd.Series(
        name_core_correct_values, dtype="boolean"
    )
    merged["name_full_correct"] = pd.Series(
        name_full_correct_values, dtype="boolean"
    )
    # 兼容旧字段：name_correct等同于核心药品名称是否正确。
    merged["name_correct"] = merged["name_core_correct"]
    merged["name_match_type"] = name_match_types
    merged["spec_correct"] = pd.Series(spec_correct_values, dtype="boolean")
    merged["manufacturer_correct"] = pd.Series(
        manufacturer_correct_values, dtype="boolean"
    )
    merged["mapped_sku_id"] = mapped_skus
    merged["sku_mapping_status"] = sku_mapping_statuses
    merged["sku_mapping_candidate_count"] = sku_candidate_counts
    merged["strict_sku_correct"] = pd.Series(
        strict_sku_correct_values, dtype="boolean"
    )

    development_count = int(
        config.get("experiment", {}).get(
            "prompt_development_sample_count",
            DEFAULT_PROMPT_DEVELOPMENT_SAMPLE_COUNT,
        )
    )
    merged["is_prompt_development_sample"] = merged["sample_id"].apply(
        lambda value: is_prompt_development_sample(value, development_count)
    )

    # 错误确定识别：模型输出matched，但应拒识，或核心药品名称错误。
    merged["overconfident_wrong_match"] = (
        (merged["status"] == "ok")
        & (merged["decision"] == "matched")
        & (
            (merged["expected_decision"] != "matched")
            | (
                (merged["expected_decision"] == "matched")
                & (merged["name_core_correct"] == False)  # noqa: E712
            )
        )
    )

    merged["subset"] = merged.apply(
        lambda row: classify_subset(row, merged_gt_cols), axis=1
    )

    error_type_lists = merged.apply(
        lambda row: determine_error_types(row, merged_gt_cols), axis=1
    )
    merged["error_types"] = error_type_lists.apply(lambda items: "|".join(items))
    merged["primary_error_type"] = error_type_lists.apply(
        lambda items: items[0] if items else ""
    )

    # Core denominators and numerators.
    gt_available_mask = merged["gt_match_status"] == "matched"
    main_mask = merged["is_main_identification_sample"] == True  # noqa: E712
    api_success_mask = merged["status"].isin(["ok", "format_error"])
    format_valid_mask = merged["status"] == "ok"

    special_mask = merged["expected_decision"].isin(
        ["unknown", "multi_object", "non_drug"]
    )
    correct_rejection_mask = special_mask & merged["decision_correct"]
    unsupported_definitive_mask = special_mask & (merged["decision"] == "matched")

    matched_with_gt_mask = (
        gt_available_mask
        & (merged["status"] == "ok")
        & (merged["decision"] == "matched")
    )
    overconfident_wrong_mask = (
        matched_with_gt_mask & merged["overconfident_wrong_match"]
    )

    core_correct_mask = merged["name_core_correct"] == True  # noqa: E712
    core_only_mask = (
        core_correct_mask & (merged["name_full_correct"] == False)  # noqa: E712
    )

    blind_mask = ~merged["is_prompt_development_sample"]
    blind_merged = merged.loc[blind_mask].copy()
    blind_gt_available_mask = blind_merged["gt_match_status"] == "matched"
    blind_special_mask = blind_merged["expected_decision"].isin(
        ["unknown", "multi_object", "non_drug"]
    )

    metrics: Dict[str, Any] = {
        "experiment": {
            "name": config["experiment"]["name"],
            "evaluated_at": now_iso(),
            "model": config["api"]["model"],
            "strict_sku_mode": config["experiment"].get("strict_sku_mode", "B"),
        },
        "sample_counts": {
            "manifest_total": int(len(merged)),
            "gt_matched": int(gt_available_mask.sum()),
            "gt_unmatched_or_ambiguous": int((~gt_available_mask).sum()),
            "main_identification_samples": int(main_mask.sum()),
            "special_rejection_samples": int(special_mask.sum()),
            "prediction_ok": int((merged["status"] == "ok").sum()),
            "format_error": int((merged["status"] == "format_error").sum()),
            "api_error": int((merged["status"] == "api_error").sum()),
            "input_error": int((merged["status"] == "input_error").sum()),
            "missing_prediction": int(merged["status"].isna().sum()),
            "prompt_development_samples": int(
                merged["is_prompt_development_sample"].sum()
            ),
            "blind_samples_excluding_prompt_development": int(blind_mask.sum()),
        },
        "name_matching_rules": {
            "core_name_allows_parenthetical_qualifier_removal": True,
            "core_name_allows_terminal_dosage_form_removal": True,
            "arbitrary_prefix_or_substring_matching": False,
            "fuzzy_edit_distance_matching": False,
            "examples": {
                "菊花(贡菊)_vs_菊花": "core_correct_full_incorrect",
                "乳癖消片_vs_乳癖消": "core_correct_full_incorrect",
            },
        },
        "recognition": recognition_metric_bundle(merged),
        "recognition_blind_excluding_prompt_development_samples": (
            recognition_metric_bundle(blind_merged)
        ),
        "decision_and_rejection": {
            "overall_decision_accuracy": metric_record(
                int((merged.loc[gt_available_mask, "decision_correct"] == True).sum()),  # noqa: E712
                int(gt_available_mask.sum()),
            ),
            "correct_rejection_rate": metric_record(
                int(correct_rejection_mask.sum()), int(special_mask.sum())
            ),
            "unsupported_definitive_answer_rate": metric_record(
                int(unsupported_definitive_mask.sum()), int(special_mask.sum())
            ),
            "overconfident_wrong_match_rate": metric_record(
                int(overconfident_wrong_mask.sum()),
                int(matched_with_gt_mask.sum()),
            ),
            "core_only_name_rate_among_core_correct": metric_record(
                int(core_only_mask.sum()),
                int(core_correct_mask.sum()),
            ),
            "blind_overall_decision_accuracy": metric_record(
                int(
                    (
                        blind_merged.loc[
                            blind_gt_available_mask, "decision_correct"
                        ]
                        == True  # noqa: E712
                    ).sum()
                ),
                int(blind_gt_available_mask.sum()),
            ),
            "blind_correct_rejection_rate": metric_record(
                int(
                    (
                        blind_special_mask
                        & blind_merged["decision_correct"]
                    ).sum()
                ),
                int(blind_special_mask.sum()),
            ),
        },
        "technical": {
            "api_success_rate": metric_record(
                int(api_success_mask.sum()), int(len(merged))
            ),
            "json_format_valid_rate": metric_record(
                int(format_valid_mask.sum()), int(api_success_mask.sum())
            ),
            "average_latency_seconds": safe_mean(
                numeric_series(merged, "latency_seconds")
            ),
            "average_prompt_tokens": safe_mean(
                numeric_series(merged, "prompt_tokens")
            ),
            "average_completion_tokens": safe_mean(
                numeric_series(merged, "completion_tokens")
            ),
            "average_total_tokens": safe_mean(numeric_series(merged, "total_tokens")),
            "total_prompt_tokens": safe_sum(numeric_series(merged, "prompt_tokens")),
            "total_completion_tokens": safe_sum(
                numeric_series(merged, "completion_tokens")
            ),
            "total_tokens": safe_sum(numeric_series(merged, "total_tokens")),
        },
        "sku_mapping": {
            "unique": int((merged["sku_mapping_status"] == "unique").sum()),
            "ambiguous": int((merged["sku_mapping_status"] == "ambiguous").sum()),
            "name_not_found": int(
                (merged["sku_mapping_status"] == "name_not_found").sum()
            ),
            "specification_not_found": int(
                (merged["sku_mapping_status"] == "specification_not_found").sum()
            ),
            "manufacturer_not_found": int(
                (merged["sku_mapping_status"] == "manufacturer_not_found").sum()
            ),
            "no_predicted_name": int(
                (merged["sku_mapping_status"] == "no_predicted_name").sum()
            ),
        },
        "subsets": {},
        "resolved_ground_truth_columns": gt_cols,
    }

    for subset_name, subset_df in merged[gt_available_mask].groupby("subset"):
        subset_metrics: Dict[str, Any] = {
            "sample_count": int(len(subset_df)),
            "decision_accuracy": metric_record(
                int((subset_df["decision_correct"] == True).sum()),  # noqa: E712
                int(len(subset_df)),
            ),
        }
        for output_name, column in [
            ("drug_name_accuracy", "name_core_correct"),
            ("drug_name_core_accuracy", "name_core_correct"),
            ("drug_name_full_accuracy", "name_full_correct"),
            ("specification_accuracy", "spec_correct"),
            ("manufacturer_accuracy", "manufacturer_correct"),
            ("strict_sku_accuracy", "strict_sku_correct"),
        ]:
            valid = subset_df[column].notna()
            subset_metrics[output_name] = metric_record(
                int((subset_df.loc[valid, column] == True).sum()),  # noqa: E712
                int(valid.sum()),
            )
        metrics["subsets"][str(subset_name)] = subset_metrics

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    error_mask = merged["error_types"].astype(str).str.len() > 0
    error_cases = merged.loc[error_mask].copy()
    error_cases.to_csv(error_cases_path, index=False, encoding="utf-8-sig")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, ensure_ascii=False, indent=2, allow_nan=False)

    print("评价完成。")
    print("合并结果：{}".format(merged_path))
    print("指标文件：{}".format(metrics_path))
    print("错误样本：{}".format(error_cases_path))

    recognition = metrics["recognition"]
    for label, key in [
        ("核心药品名称准确率", "drug_name_core_accuracy"),
        ("完整商品名称准确率", "drug_name_full_accuracy"),
        ("规格准确率", "specification_accuracy"),
        ("厂家准确率", "manufacturer_accuracy"),
        ("严格SKU准确率", "strict_sku_accuracy"),
    ]:
        item = recognition[key]
        value = item["value"]
        display = "N/A" if value is None else "{:.2%}".format(value)
        print(
            "{}：{}（{}/{}）".format(
                label, display, item["numerator"], item["denominator"]
            )
        )

    blind_recognition = metrics[
        "recognition_blind_excluding_prompt_development_samples"
    ]
    blind_core = blind_recognition["drug_name_core_accuracy"]
    blind_core_value = blind_core["value"]
    print(
        "排除前{}个提示词开发样本后的核心药名准确率：{}（{}/{}）".format(
            development_count,
            "N/A"
            if blind_core_value is None
            else "{:.2%}".format(blind_core_value),
            blind_core["numerator"],
            blind_core["denominator"],
        )
    )

    overconfident = metrics["decision_and_rejection"][
        "overconfident_wrong_match_rate"
    ]
    overconfident_value = overconfident["value"]
    print(
        "错误确定识别率：{}（{}/{}）".format(
            "N/A"
            if overconfident_value is None
            else "{:.2%}".format(overconfident_value),
            overconfident["numerator"],
            overconfident["denominator"],
        )
    )

    rejection = metrics["decision_and_rejection"]["correct_rejection_rate"]
    rejection_value = rejection["value"]
    print(
        "正确拒识率：{}（{}/{}）".format(
            "N/A" if rejection_value is None else "{:.2%}".format(rejection_value),
            rejection["numerator"],
            rejection["denominator"],
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("\n评价失败：{}".format(error), file=sys.stderr)
        sys.exit(1)
