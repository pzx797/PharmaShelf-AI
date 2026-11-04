# -*- coding: utf-8 -*-
"""运行Qwen3.7-Plus药品裁剪图RAG识别实验。

运行顺序：
    1. python 01_prepare_rag_experiment.py
    2. python 02_build_rag_index.py
    3. python 03_run_qwen37plus_rag.py --limit 3
    4. python 03_run_qwen37plus_rag.py
    5. python 04_evaluate_rag_results.py
    6. python 05_evaluate_retrieval_results.py

核心流程：
- 读取已冻结的直接基线视觉输出；
- 过滤货架标签、价格标签、门店标识等非商品包装文字；
- 根据清洗后的名称、规格、厂家、条码、可见文字和包装特征构造检索查询；
- 从本地知识库索引中召回Top-K候选；
- 将裁剪图、第一阶段视觉线索和候选记录交给Qwen复核；
- 若模型选中候选SKU，则从知识库确定性回填最终结构化字段；
- 支持断点续跑、API重试、原始响应和检索候选留档。

该脚本不读取人工真值表。

依赖：
    pip install -U openai pandas pyyaml scikit-learn joblib

兼容：Python 3.8+
"""

from __future__ import print_function

import argparse
import base64
import json
import math
import mimetypes
import os
import random
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import joblib
    import numpy as np
    import pandas as pd
    import yaml
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit(
        "缺少依赖。请先执行："
        "pip install -U openai pandas pyyaml scikit-learn joblib numpy\n"
        "原始错误：{}".format(exc)
    )


OUTPUT_ROOT = Path(r"E:\EdgeDownload\ultralytics-main\qwen37plus_rag_experiment")
RUN_NAME = "qwen37plus_rag_v1"
DEFAULT_CONFIG_PATH = OUTPUT_ROOT / "runs" / RUN_NAME / "config.yaml"

ALLOWED_DECISIONS = {"matched", "unknown", "multi_object", "non_drug"}
ALLOWED_CONSISTENCY = {"strong", "partial", "none"}

# 以下关键词表示文字来源可能不是商品包装本身，不能直接用于商品身份检索。
NON_PRODUCT_CONTEXT_KEYWORDS = (
    "货架标签",
    "货架价签",
    "价格标签",
    "价格牌",
    "价签",
    "价牌",
    "门店标签",
    "药店标签",
    "下方标签",
    "上方标签",
    "旁边标签",
    "侧边标签",
    "背景文字",
    "货架文字",
)

# 常见门店、价格和促销文本。它们即使出现在visible_text中，也不应作为SKU检索证据。
NON_PRODUCT_TEXT_PATTERNS = (
    r"^健之佳$",
    r"^一心堂$",
    r"^老百姓大药房$",
    r"^益丰大药房$",
    r"^大参林$",
    r"^海王星辰$",

    # 明确带有货币符号的价格。
    r"^[¥￥]\d+(?:\.\d{1,2})?$",

    # 明确带有“元”的价格。
    r"^\d+(?:\.\d{1,2})?元$",

    # 带有价格语义前缀的文字。
    r"^(?:会员价|零售价|促销价|活动价|售价|价格)[：:]?[¥￥]?\d+(?:\.\d{1,2})?元?$",

    r"^(会员价|零售价|促销价|活动价|售价|价格)$",
    r"^(会员|促销|特价|活动|扫码购)$",
)

# 前25列与直接基线评价脚本兼容；后续字段用于RAG诊断。
PREDICTION_COLUMNS = [
    "run_id",
    "sample_id",
    "status",
    "decision",
    "pred_drug_name",
    "pred_generic_name",
    "pred_specification",
    "pred_manufacturer",
    "pred_barcode_or_approval_no",
    "pred_sku_id",
    "confidence",
    "visible_text_json",
    "package_features_json",
    "reason",
    "model",
    "response_id",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "latency_seconds",
    "attempt_count",
    "processed_at",
    "parse_error",
    "direct_status",
    "direct_decision",
    "original_direct_visual_result_json",
    "sanitized_direct_visual_result_json",
    "retrieval_filter_applied",
    "removed_visible_text_json",
    "retrieval_filter_reasons_json",
    "retrieval_query",
    "retrieval_method",
    "retrieved_candidate_count",
    "retrieved_candidates_json",
    "retrieval_top1_sku_id",
    "retrieval_top1_score",
    "selected_candidate_rank",
    "candidate_consistency",
    "evidence_from_image_json",
    "evidence_from_kb_json",
    "model_pred_drug_name",
    "model_pred_generic_name",
    "model_pred_specification",
    "model_pred_manufacturer",
    "model_pred_barcode_or_approval_no",
    "model_pred_sku_id",
    "context_guard_triggered",
    "context_guard_reason",
]


# -----------------------------------------------------------------------------
# 文件与文本辅助函数
# -----------------------------------------------------------------------------
def now_iso() -> str:
    """返回带本地时区的ISO时间字符串。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_yaml(path: Path) -> Dict[str, Any]:
    """读取YAML配置文件。"""
    if not path.exists():
        raise FileNotFoundError(
            "未找到配置文件：{}。请先运行01_prepare_rag_experiment.py".format(path)
        )
    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    if not isinstance(config, dict):
        raise ValueError("config.yaml内容无效")
    return config


def read_text(path: Path) -> str:
    """读取UTF-8文本文件。"""
    if not path.exists():
        raise FileNotFoundError("未找到文本文件：{}".format(path))
    return path.read_text(encoding="utf-8").strip()


def append_line(path: Path, text: str) -> None:
    """向日志文件追加一行文本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(text.rstrip() + "\n")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """向JSONL文件追加一条JSON记录。"""
    append_line(path, json.dumps(payload, ensure_ascii=False))


def safe_json_value(value: Any) -> Any:
    """将NumPy和pandas标量转换为可JSON序列化的Python值。"""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def clean_cell(value: Any) -> str:
    """将表格值转换为稳定字符串，避免SKU和条码出现小数后缀。"""
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


def normalize_text(value: Any) -> str:
    """标准化文本，用于字段精确匹配和包含匹配。"""
    text = clean_cell(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("×", "x").replace("✕", "x").replace("＊", "x")
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r"[，,。.;；:：'\"“”‘’/\\_\-·•()（）\[\]【】{}]+", "", text)
    return text


def normalize_sku(value: Any) -> str:
    """标准化SKU编号。"""
    return normalize_text(value)


def image_to_data_url(image_path: Path) -> str:
    """将本地图片转换为API可接受的Base64 data URL。"""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        suffix_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = suffix_map.get(image_path.suffix.lower())
    if not mime_type:
        raise ValueError("无法确定图片MIME类型：{}".format(image_path))
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return "data:{};base64,{}".format(mime_type, encoded)


# -----------------------------------------------------------------------------
# JSON响应解析与字段校验
# -----------------------------------------------------------------------------
def strip_code_fence(text: str) -> str:
    """移除模型偶尔返回的Markdown代码围栏。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_first_json_object(text: str) -> str:
    """从混合文本中提取首个完整JSON对象。"""
    start = text.find("{")
    if start < 0:
        raise ValueError("响应中未找到JSON对象")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("JSON对象括号不完整")


def parse_json_answer(content: str) -> Dict[str, Any]:
    """解析模型返回的JSON对象。"""
    cleaned = strip_code_fence(content)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = json.loads(extract_first_json_object(cleaned))
    if not isinstance(value, dict):
        raise ValueError("模型响应不是JSON对象")
    return value


def nullable_text(value: Any) -> Optional[str]:
    """将空值、null文本和NaN统一为None。"""
    text = clean_cell(value)
    if not text:
        return None
    if text.casefold() in {"null", "none", "nan"}:
        return None
    return text


def normalize_string_list(value: Any) -> List[str]:
    """将模型字段规范为字符串列表。"""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = []
        for item in value:
            text = nullable_text(item)
            if text:
                result.append(text)
        return result
    raise ValueError("应为字符串列表，实际为{}".format(type(value).__name__))


def normalize_confidence(value: Any) -> Optional[float]:
    """将模型置信度统一规范到0到1范围。

    兼容以下常见输出：
    - 0.92
    - 92
    - "92%"

    空值保留为None；无法转换的内容仍抛出异常，避免静默吞掉真正的格式问题。
    """
    if value is None or clean_cell(value) == "":
        return None

    text = clean_cell(value).replace("%", "").strip()
    number = float(text)

    # 模型偶尔按百分制返回置信度，例如92或92%。
    if 1.0 < number <= 100.0:
        number /= 100.0

    if number < 0.0 or number > 1.0:
        raise ValueError("confidence必须在0到1之间，实际为：{}".format(value))

    return round(number, 4)


def normalize_candidate_rank(value: Any) -> Optional[int]:
    """将候选排名规范为正整数或None。"""
    if value is None or clean_cell(value) == "":
        return None
    rank = int(float(value))
    if rank < 1:
        raise ValueError("selected_candidate_rank必须为正整数")
    return rank


def normalize_candidate_consistency(
    value: Any,
    pred_sku_id: Optional[str],
) -> str:
    """规范候选一致性字段。

    处理原则：
    1.兼容模型返回的中英文常见表达；
    2.未选择SKU且字段为空时，规范为none；
    3.已选择SKU但字段为空时，保守规范为partial；
    4.无法识别的非空内容继续报错，避免掩盖真实异常。

    candidate_consistency属于RAG诊断字段。对空值进行规范化不会修改
    decision、pred_sku_id、商品名称等主要识别结果。
    """
    text = clean_cell(value).casefold()

    mapping = {
        # 高度一致
        "strong": "strong",
        "完全一致": "strong",
        "高度一致": "strong",
        "强一致": "strong",
        "一致性强": "strong",

        # 部分一致或证据不完整
        "partial": "partial",
        "部分一致": "partial",
        "基本一致": "partial",
        "弱一致": "partial",
        "一致性一般": "partial",
        "weak": "partial",

        # 不一致、未选候选或无法形成候选支持
        "none": "none",
        "不一致": "none",
        "无一致候选": "none",
        "没有一致候选": "none",
        "无": "none",
        "无法判断": "none",
        "unknown": "none",
    }

    if text in mapping:
        return mapping[text]

    if not text and not pred_sku_id:
        return "none"

    if not text and pred_sku_id:
        # 已选择SKU但漏填一致性时，不自动判为strong，保守记为partial。
        return "partial"

    raise ValueError("candidate_consistency无效：{}".format(value))


def validate_prediction(
    parsed: Dict[str, Any],
    sample_id: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """校验RAG模型输出，并限制SKU只能来自检索候选。"""
    required = {
        "sample_id",
        "decision",
        "pred_drug_name",
        "pred_generic_name",
        "pred_specification",
        "pred_manufacturer",
        "pred_barcode_or_approval_no",
        "pred_sku_id",
        "selected_candidate_rank",
        "candidate_consistency",
        "confidence",
        "visible_text",
        "package_features",
        "evidence_from_image",
        "evidence_from_kb",
        "reason",
    }
    missing = required - set(parsed.keys())
    if missing:
        raise KeyError("模型JSON缺少字段：{}".format(sorted(missing)))

    returned_sample_id = clean_cell(parsed.get("sample_id"))
    if returned_sample_id and returned_sample_id != sample_id:
        raise ValueError(
            "sample_id不一致：期望{}，实际{}".format(sample_id, returned_sample_id)
        )

    decision = clean_cell(parsed.get("decision")).casefold()
    if decision not in ALLOWED_DECISIONS:
        raise ValueError("decision无效：{}".format(decision))

    pred_sku_id = nullable_text(parsed.get("pred_sku_id"))
    consistency = normalize_candidate_consistency(
        parsed.get("candidate_consistency"),
        pred_sku_id,
    )
    candidate_map = {
        normalize_sku(candidate.get("sku_id")): candidate for candidate in candidates
    }
    if pred_sku_id:
        normalized_pred_sku = normalize_sku(pred_sku_id)
        if normalized_pred_sku not in candidate_map:
            raise ValueError(
                "pred_sku_id不在当前Top-K候选中：{}".format(pred_sku_id)
            )
        pred_sku_id = clean_cell(candidate_map[normalized_pred_sku].get("sku_id"))

    rank = normalize_candidate_rank(parsed.get("selected_candidate_rank"))
    if pred_sku_id:
        actual_rank = next(
            int(candidate["rank"])
            for candidate in candidates
            if normalize_sku(candidate.get("sku_id")) == normalize_sku(pred_sku_id)
        )
        rank = actual_rank
    elif rank is not None:
        # 未选择SKU时不保留无意义排名。
        rank = None

    return {
        "sample_id": sample_id,
        "decision": decision,
        "pred_drug_name": nullable_text(parsed.get("pred_drug_name")),
        "pred_generic_name": nullable_text(parsed.get("pred_generic_name")),
        "pred_specification": nullable_text(parsed.get("pred_specification")),
        "pred_manufacturer": nullable_text(parsed.get("pred_manufacturer")),
        "pred_barcode_or_approval_no": nullable_text(
            parsed.get("pred_barcode_or_approval_no")
        ),
        "pred_sku_id": pred_sku_id,
        "selected_candidate_rank": rank,
        "candidate_consistency": consistency,
        "confidence": normalize_confidence(parsed.get("confidence")),
        "visible_text": normalize_string_list(parsed.get("visible_text")),
        "package_features": normalize_string_list(parsed.get("package_features")),
        "evidence_from_image": normalize_string_list(
            parsed.get("evidence_from_image")
        ),
        "evidence_from_kb": normalize_string_list(parsed.get("evidence_from_kb")),
        "reason": nullable_text(parsed.get("reason")),
    }


# -----------------------------------------------------------------------------
# 直接基线视觉线索与RAG查询构造
# -----------------------------------------------------------------------------
def parse_json_list_cell(value: Any) -> List[str]:
    """解析CSV中的JSON字符串列表，解析失败时退化为空列表。"""
    text = clean_cell(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [clean_cell(item) for item in parsed if clean_cell(item)]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return []


def direct_visual_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    """整理第一阶段直接识别结果，供检索和最终复核使用。"""
    return {
        "status": clean_cell(row.get("status")),
        "decision": clean_cell(row.get("decision")),
        "pred_drug_name": nullable_text(row.get("pred_drug_name")),
        "pred_generic_name": nullable_text(row.get("pred_generic_name")),
        "pred_specification": nullable_text(row.get("pred_specification")),
        "pred_manufacturer": nullable_text(row.get("pred_manufacturer")),
        "pred_barcode_or_approval_no": nullable_text(
            row.get("pred_barcode_or_approval_no")
        ),
        "visible_text": parse_json_list_cell(row.get("visible_text_json")),
        "package_features": parse_json_list_cell(row.get("package_features_json")),
        "reason": nullable_text(row.get("reason")),
    }


def contains_non_product_context(text: Any) -> bool:
    """判断文本中是否明确提到货架标签、价格标签等非商品包装来源。"""
    normalized = clean_cell(text)
    return any(keyword in normalized for keyword in NON_PRODUCT_CONTEXT_KEYWORDS)


def is_obvious_non_product_text(text: Any) -> bool:
    """识别明显的门店名、价格和促销文字。"""
    value = clean_cell(text)
    if not value:
        return True
    compact = re.sub(r"\s+", "", value)
    return any(re.fullmatch(pattern, compact, flags=re.I) for pattern in NON_PRODUCT_TEXT_PATTERNS)


def extract_context_text_from_reason(reason: Any) -> List[str]:
    """从第一阶段reason中提取被明确描述为货架标签或价格标签的文字。

    例如：
        货架标签清晰显示“百合干”
        下方价签写有'菊花'

    返回值只用于过滤检索线索，不参与模型推理真值。
    """
    text = clean_cell(reason)
    if not text or not contains_non_product_context(text):
        return []

    results: List[str] = []
    source_pattern = "(?:{})".format(
        "|".join(re.escape(keyword) for keyword in NON_PRODUCT_CONTEXT_KEYWORDS)
    )

    # 优先提取引号中的文字。
    quoted_patterns = [
        source_pattern + r"[^。；;]{0,24}?[“\"]([^”\"]{1,40})[”\"]",
        source_pattern + r"[^。；;]{0,24}?[‘']([^’']{1,40})[’']",
    ]
    for pattern in quoted_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            value = clean_cell(match.group(1))
            if value:
                results.append(value)

    # 兼容没有引号的短表达，例如“货架标签显示百合干”。
    plain_pattern = (
        source_pattern
        + r"[^，,。；;：:]{0,12}?(?:显示|写有|标有|可见|内容为|为|是)"
        + r"[：:\s]*([^，,。；;]{1,20})"
    )
    for match in re.finditer(plain_pattern, text, flags=re.I):
        value = clean_cell(match.group(1))
        value = re.sub(r"^(?:文字|内容)[为是：:\s]*", "", value)
        if value:
            results.append(value)

    # 去重并保留原顺序。
    unique: List[str] = []
    seen = set()
    for item in results:
        key = normalize_text(item)
        if key and key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def text_matches_any(value: Any, candidates: Sequence[Any]) -> bool:
    """判断一个文本与候选文本是否相同或存在明显包含关系。"""
    normalized_value = normalize_text(value)
    if not normalized_value:
        return False
    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)
        if not normalized_candidate:
            continue
        if (
            normalized_value == normalized_candidate
            or normalized_value in normalized_candidate
            or normalized_candidate in normalized_value
        ):
            return True
    return False


def sanitize_direct_payload_for_retrieval(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """过滤第一阶段结果中的非商品包装线索。

    该函数只改变用于检索和第二阶段提示词的副本，原始Direct结果会完整保存在
    retrieval_results.jsonl和raw_responses.jsonl中，便于实验审计。

    过滤原则：
    1.移除reason中明确归属于货架标签、价格标签等来源的visible_text；
    2.移除明显的门店名、价格和促销文字；
    3.移除描述货架标签、价格标签和背景区域的package_features；
    4.若身份字段与被移除文字相同，则同步清空该字段；
    5.不读取人工真值，不根据知识库反向修改第一阶段结果。
    """
    sanitized: Dict[str, Any] = {
        "status": clean_cell(payload.get("status")),
        "decision": clean_cell(payload.get("decision")),
        "pred_drug_name": nullable_text(payload.get("pred_drug_name")),
        "pred_generic_name": nullable_text(payload.get("pred_generic_name")),
        "pred_specification": nullable_text(payload.get("pred_specification")),
        "pred_manufacturer": nullable_text(payload.get("pred_manufacturer")),
        "pred_barcode_or_approval_no": nullable_text(
            payload.get("pred_barcode_or_approval_no")
        ),
        "visible_text": list(payload.get("visible_text", []) or []),
        "package_features": list(payload.get("package_features", []) or []),
        "reason": nullable_text(payload.get("reason")),
    }

    original_reason = clean_cell(payload.get("reason"))
    context_in_reason = contains_non_product_context(original_reason)
    reason_context_text = extract_context_text_from_reason(original_reason)

    removed_visible_text: List[str] = []
    kept_visible_text: List[str] = []
    for item in sanitized["visible_text"]:
        should_remove = (
            is_obvious_non_product_text(item)
            or text_matches_any(item, reason_context_text)
        )
        if should_remove:
            removed_visible_text.append(clean_cell(item))
        else:
            kept_visible_text.append(clean_cell(item))
    sanitized["visible_text"] = [item for item in kept_visible_text if item]

    removed_package_features: List[str] = []
    kept_package_features: List[str] = []
    for item in sanitized["package_features"]:
        if contains_non_product_context(item) or any(
            keyword in clean_cell(item)
            for keyword in ("货架", "价签", "价格牌", "背景区域", "门店标识")
        ):
            removed_package_features.append(clean_cell(item))
        else:
            kept_package_features.append(clean_cell(item))
    sanitized["package_features"] = [
        item for item in kept_package_features if item
    ]

    removed_identity_fields: List[str] = []
    identity_fields = (
        "pred_drug_name",
        "pred_generic_name",
        "pred_specification",
        "pred_manufacturer",
        "pred_barcode_or_approval_no",
    )
    removed_identity_text = removed_visible_text + reason_context_text
    for field in identity_fields:
        value = sanitized.get(field)
        if value and text_matches_any(value, removed_identity_text):
            sanitized[field] = None
            removed_identity_fields.append(field)

    filter_reasons: List[str] = []
    if context_in_reason:
        filter_reasons.append("direct_reason_mentions_non_product_context")
    if removed_visible_text:
        filter_reasons.append("removed_non_product_visible_text")
    if removed_package_features:
        filter_reasons.append("removed_non_product_package_features")
    if removed_identity_fields:
        filter_reasons.append("cleared_identity_fields_derived_from_context_text")

    filter_applied = bool(filter_reasons)
    if filter_applied:
        sanitized["reason"] = (
            "第一阶段结果包含可能来自货架标签、价格标签、门店标识或背景的文字，"
            "检索阶段已排除这些非商品包装线索。不得依据被排除文字确定商品名称或SKU。"
        )
        sanitized["retrieval_filter_applied"] = True
        sanitized["excluded_non_product_context_text"] = removed_visible_text
        sanitized["retrieval_filter_note"] = filter_reasons
    else:
        sanitized["retrieval_filter_applied"] = False
        sanitized["excluded_non_product_context_text"] = []
        sanitized["retrieval_filter_note"] = []

    audit = {
        "filter_applied": filter_applied,
        "filter_reasons": filter_reasons,
        "reason_context_text": reason_context_text,
        "removed_visible_text": removed_visible_text,
        "removed_package_features": removed_package_features,
        "removed_identity_fields": removed_identity_fields,
    }
    return sanitized, audit


def build_retrieval_query(payload: Dict[str, Any]) -> str:
    """按身份字段权重构造知识库检索查询文本。"""
    drug_name = clean_cell(payload.get("pred_drug_name"))
    generic_name = clean_cell(payload.get("pred_generic_name"))
    specification = clean_cell(payload.get("pred_specification"))
    manufacturer = clean_cell(payload.get("pred_manufacturer"))
    barcode = clean_cell(payload.get("pred_barcode_or_approval_no"))
    visible_text = [clean_cell(item) for item in payload.get("visible_text", [])]
    package_features = [
        clean_cell(item) for item in payload.get("package_features", [])
    ]

    # 名称、条码和可见文字重复加入，提高字符检索的权重。
    parts = [
        drug_name,
        drug_name,
        drug_name,
        generic_name,
        generic_name,
        specification,
        specification,
        manufacturer,
        manufacturer,
        barcode,
        barcode,
        barcode,
    ]
    parts.extend(visible_text)
    parts.extend(visible_text)
    parts.extend(package_features)
    return " ".join(part for part in parts if part).strip()


# -----------------------------------------------------------------------------
# 检索评分与候选构造
# -----------------------------------------------------------------------------
def prepare_query_norms(payload: Dict[str, Any]) -> Dict[str, Any]:
    """预先标准化一次查询字段，避免对每条知识库记录重复处理。"""
    return {
        "barcode": normalize_text(payload.get("pred_barcode_or_approval_no")),
        "drug_name": normalize_text(payload.get("pred_drug_name")),
        "generic_name": normalize_text(payload.get("pred_generic_name")),
        "specification": normalize_text(payload.get("pred_specification")),
        "manufacturer": normalize_text(payload.get("pred_manufacturer")),
        "visible_text": [
            normalize_text(item)
            for item in payload.get("visible_text", [])
            if normalize_text(item)
        ],
    }


def exact_field_boost(
    query_norms: Dict[str, Any],
    record: Dict[str, Any],
    boosts: Dict[str, float],
) -> Tuple[float, List[str]]:
    """根据名称、规格、厂家和条码的一致性计算精确匹配加权。"""
    score = 0.0
    evidence: List[str] = []

    query_barcode = clean_cell(query_norms.get("barcode"))
    record_barcode = clean_cell(record.get("norm_barcode"))
    if query_barcode and record_barcode and query_barcode == record_barcode:
        score += float(boosts.get("barcode_exact", 1.20))
        evidence.append("barcode_exact")

    query_drug = clean_cell(query_norms.get("drug_name"))
    record_drug = clean_cell(record.get("norm_drug_name"))
    if query_drug and record_drug:
        if query_drug == record_drug:
            score += float(boosts.get("drug_name_exact", 0.85))
            evidence.append("drug_name_exact")
        elif query_drug in record_drug or record_drug in query_drug:
            score += float(boosts.get("drug_name_contains", 0.35))
            evidence.append("drug_name_contains")

    query_generic = clean_cell(query_norms.get("generic_name"))
    record_generic = clean_cell(record.get("norm_generic_name"))
    if query_generic and record_generic and query_generic == record_generic:
        score += float(boosts.get("generic_name_exact", 0.75))
        evidence.append("generic_name_exact")

    query_spec = clean_cell(query_norms.get("specification"))
    record_spec = clean_cell(record.get("norm_specification"))
    if query_spec and record_spec and query_spec == record_spec:
        score += float(boosts.get("specification_exact", 0.30))
        evidence.append("specification_exact")

    query_manufacturer = clean_cell(query_norms.get("manufacturer"))
    record_manufacturer = clean_cell(record.get("norm_manufacturer"))
    if (
        query_manufacturer
        and record_manufacturer
        and query_manufacturer == record_manufacturer
    ):
        score += float(boosts.get("manufacturer_exact", 0.20))
        evidence.append("manufacturer_exact")

    aliases = clean_cell(record.get("norm_aliases"))
    visible_values = query_norms.get("visible_text", [])
    alias_hit = any(value and value in aliases for value in visible_values)
    if aliases and alias_hit:
        score += float(boosts.get("alias_exact", 0.35))
        evidence.append("visible_text_alias_hit")

    return score, evidence

def retrieve_candidates(
    query_text: str,
    payload: Dict[str, Any],
    bundle: Dict[str, Any],
    top_k: int,
    boosts: Dict[str, float],
) -> List[Dict[str, Any]]:
    """执行TF-IDF检索和字段加权，返回排序后的Top-K候选。"""
    if not query_text.strip():
        return []

    vectorizer = bundle["vectorizer"]
    matrix = bundle["matrix"]
    records = bundle["records"]

    query_vector = vectorizer.transform([query_text])
    if query_vector.nnz == 0:
        base_scores = np.zeros(matrix.shape[0], dtype=np.float32)
    else:
        # TF-IDF已L2归一化，点积即余弦相似度。
        base_scores = (query_vector @ matrix.T).toarray().ravel()

    query_norms = prepare_query_norms(payload)
    scored: List[Tuple[float, float, float, int, List[str]]] = []
    for index, record in enumerate(records):
        field_score, evidence = exact_field_boost(query_norms, record, boosts)
        base_score = float(base_scores[index])
        final_score = base_score + field_score
        scored.append((final_score, base_score, field_score, index, evidence))

    scored.sort(key=lambda item: (-item[0], -item[1], item[3]))
    selected = scored[: max(0, int(top_k))]

    candidates: List[Dict[str, Any]] = []
    for rank, (final_score, base_score, field_score, index, evidence) in enumerate(
        selected, start=1
    ):
        record = records[index]
        candidates.append(
            {
                "rank": rank,
                "sku_id": clean_cell(record.get("sku_id")),
                "drug_name": nullable_text(record.get("drug_name")),
                "generic_name": nullable_text(record.get("generic_name")),
                "specification": nullable_text(record.get("specification")),
                "manufacturer": nullable_text(record.get("manufacturer")),
                "barcode": nullable_text(record.get("barcode")),
                "category": nullable_text(record.get("category")),
                "aliases": nullable_text(record.get("aliases")),
                "visual_keywords": nullable_text(record.get("visual_keywords")),
                "retrieval_score": round(float(final_score), 6),
                "tfidf_score": round(float(base_score), 6),
                "field_boost_score": round(float(field_score), 6),
                "retrieval_evidence": evidence,
            }
        )
    return candidates


def candidate_prompt_payload(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """生成精简候选上下文，避免把内部检索字段全部发送给模型。"""
    fields = [
        "rank",
        "sku_id",
        "drug_name",
        "generic_name",
        "specification",
        "manufacturer",
        "barcode",
        "category",
        "aliases",
        "visual_keywords",
        "retrieval_score",
    ]
    return [
        {field: candidate.get(field) for field in fields}
        for candidate in candidates
    ]


def apply_non_product_context_guard(
    validated: Dict[str, Any],
    filter_audit: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """审计模型是否重新使用了已过滤的非商品上下文。

    本函数只记录潜在风险，不再修改模型的decision、SKU或其他预测字段。

    这样做的原因：
    1.检索阶段已经过滤货架标签、价格标签和门店标识；
    2.模型可能在visible_text或解释中客观提到这些文字，
      但不代表模型依赖这些文字确定商品身份；
    3.自动将结果降级为unknown容易误伤有充分独立证据的正确结果；
    4.保留模型原始输出，更适合后续进行客观实验评价。
    """
    result = dict(validated)

    evidence_items = list(
        result.get("evidence_from_image", []) or []
    )
    reason = clean_cell(result.get("reason"))

    combined_text = " ".join(
        evidence_items + ([reason] if reason else [])
    )

    # 检查模型解释中是否出现货架标签、价格标签等语义。
    source_violation = contains_non_product_context(
        combined_text
    )

    # 检查模型是否重新提到了检索阶段删除的文字。
    removed_text = (
        filter_audit.get("removed_visible_text", []) or []
    )

    removed_text_reused = any(
        text_matches_any(item, [combined_text])
        for item in removed_text
    )

    audit_reasons: List[str] = []

    if source_violation:
        audit_reasons.append(
            "model_mentioned_non_product_context"
        )

    if removed_text_reused:
        audit_reasons.append(
            "model_rementioned_filtered_context_text"
        )

    # 仅记录风险，不修改模型结果。
    audit = {
        "triggered": False,
        "reasons": audit_reasons,
        "source_violation": source_violation,
        "removed_text_reused": removed_text_reused,
    }

    return result, audit


def canonicalize_selected_candidate(
    validated: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """若模型选择候选SKU，则用知识库记录确定性回填最终字段。"""
    result = dict(validated)
    selected_sku = normalize_sku(validated.get("pred_sku_id"))
    if not selected_sku:
        return result

    selected = next(
        candidate
        for candidate in candidates
        if normalize_sku(candidate.get("sku_id")) == selected_sku
    )
    result["pred_sku_id"] = clean_cell(selected.get("sku_id"))
    result["pred_drug_name"] = nullable_text(selected.get("drug_name"))
    result["pred_generic_name"] = nullable_text(selected.get("generic_name"))
    result["pred_specification"] = nullable_text(selected.get("specification"))
    result["pred_manufacturer"] = nullable_text(selected.get("manufacturer"))
    result["pred_barcode_or_approval_no"] = nullable_text(selected.get("barcode"))
    result["selected_candidate_rank"] = int(selected.get("rank"))
    return result


# -----------------------------------------------------------------------------
# 预测结果读写与API统计
# -----------------------------------------------------------------------------
def usage_values(
    completion: Any,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """提取API返回的token统计。"""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None, None, None, None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    reasoning_tokens = None
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
    return prompt_tokens, completion_tokens, total_tokens, reasoning_tokens


def read_existing_predictions(path: Path) -> pd.DataFrame:
    """读取已有RAG预测表并补齐新版本字段。"""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for column in PREDICTION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[PREDICTION_COLUMNS]


def write_predictions_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    """使用临时文件原子写入CSV，降低中断导致文件损坏的风险。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(str(temp_path), str(path))


def completed_sample_ids(frame: pd.DataFrame) -> set:
    """返回已完成或已确认格式失败的样本编号。"""
    if frame.empty:
        return set()
    final_statuses = {"ok", "format_error", "input_error"}
    mask = frame["status"].isin(final_statuses)
    return set(frame.loc[mask, "sample_id"].astype(str))


def upsert_prediction(rows: List[Dict[str, Any]], new_row: Dict[str, Any]) -> None:
    """按sample_id更新或插入单条预测记录。"""
    sample_id = str(new_row["sample_id"])
    for index, row in enumerate(rows):
        if str(row.get("sample_id", "")) == sample_id:
            rows[index] = new_row
            return
    rows.append(new_row)


def blank_prediction_row(run_id: str, sample_id: str, model: str) -> Dict[str, Any]:
    """创建字段完整的空预测行。"""
    row = {column: "" for column in PREDICTION_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "sample_id": sample_id,
            "model": model,
            "processed_at": now_iso(),
        }
    )
    return row


def get_retry_delay(attempt: int) -> float:
    """计算带微小随机抖动的指数退避等待时间。"""
    return min(30.0, (2.0 ** attempt) + random.uniform(0.0, 0.5))


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main() -> int:
    """执行全量RAG检索、API复核和结果保存。"""
    parser = argparse.ArgumentParser(description="运行Qwen3.7-Plus RAG识别实验")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="config.yaml路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前N个待运行样本；0表示全部。建议先用--limit 3试跑。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新运行已完成样本。正式实验中不要随意使用。",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_yaml(config_path)
    paths = config["paths"]
    api_cfg = config["api"]
    retrieval_cfg = config["retrieval"]

    manifest_path = Path(paths["manifest_path"])
    direct_predictions_path = Path(paths["direct_predictions_path"])
    index_path = Path(paths["retrieval_index_path"])
    predictions_path = Path(paths["predictions_path"])
    retrieval_results_path = Path(paths["retrieval_results_path"])
    raw_responses_path = Path(paths["raw_responses_path"])
    api_errors_log = Path(paths["api_errors_log"])
    run_info_log = Path(paths["run_info_log"])
    system_prompt_path = Path(paths["system_prompt_path"])
    user_prompt_path = Path(paths["user_prompt_path"])

    for label, path in [
        ("输入清单", manifest_path),
        ("直接基线预测", direct_predictions_path),
        ("RAG索引", index_path),
        ("系统提示词", system_prompt_path),
        ("用户提示词", user_prompt_path),
    ]:
        if not path.exists():
            raise FileNotFoundError("未找到{}：{}".format(label, path))

    manifest = pd.read_csv(
        manifest_path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    direct_frame = pd.read_csv(
        direct_predictions_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if manifest["sample_id"].duplicated().any():
        raise ValueError("输入清单存在重复sample_id")
    if direct_frame["sample_id"].duplicated().any():
        raise ValueError("直接基线预测存在重复sample_id")

    direct_by_id = {
        str(row["sample_id"]): row
        for row in direct_frame.to_dict(orient="records")
    }
    manifest = manifest[
        manifest["include_in_inference"].astype(str) != "0"
    ].copy()
    manifest = manifest.sort_values("sample_id").reset_index(drop=True)

    bundle = joblib.load(index_path)
    required_bundle_keys = {"vectorizer", "matrix", "records", "method"}
    missing_bundle_keys = required_bundle_keys - set(bundle.keys())
    if missing_bundle_keys:
        raise KeyError("RAG索引缺少内容：{}".format(sorted(missing_bundle_keys)))

    system_prompt = read_text(system_prompt_path)
    user_prompt_template = read_text(user_prompt_path)

    api_key_env = str(api_cfg["api_key_env"])
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise EnvironmentError(
            "未读取到环境变量{}。PowerShell临时设置示例：\n"
            "$env:{}=\"你的API-Key\"".format(api_key_env, api_key_env)
        )

    client = OpenAI(
        api_key=api_key,
        base_url=str(api_cfg["base_url"]),
        timeout=float(api_cfg.get("timeout_seconds", 300)),
        max_retries=0,
    )

    model = str(api_cfg["model"])
    enable_thinking = bool(api_cfg.get("enable_thinking", True))
    temperature = float(api_cfg.get("temperature", 0.1))
    max_tokens = int(api_cfg.get("max_tokens", 1024))
    max_retries = int(api_cfg.get("max_retries", 3))
    vl_high_resolution_images = bool(
        api_cfg.get("vl_high_resolution_images", True)
    )
    top_k = int(retrieval_cfg.get("top_k", 5))
    field_boosts = {
        key: float(value)
        for key, value in retrieval_cfg.get("field_boosts", {}).items()
    }
    retrieval_method = str(retrieval_cfg.get("method", bundle.get("method", "")))

    existing = read_existing_predictions(predictions_path)
    prediction_rows: List[Dict[str, Any]] = existing.to_dict(orient="records")
    completed = set() if args.force else completed_sample_ids(existing)

    pending = manifest[~manifest["sample_id"].isin(completed)].copy()
    if args.limit and args.limit > 0:
        pending = pending.head(args.limit)

    run_id = "rag_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_metadata = {
        "run_id": run_id,
        "started_at": now_iso(),
        "config_path": str(config_path),
        "model": model,
        "base_url": str(api_cfg["base_url"]),
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "vl_high_resolution_images": vl_high_resolution_images,
        "retrieval_method": retrieval_method,
        "top_k": top_k,
        "query_source": retrieval_cfg.get("query_source"),
        "ground_truth_used": False,
        "manifest_total": int(len(manifest)),
        "already_completed": int(len(completed)),
        "pending_this_run": int(len(pending)),
        "force": bool(args.force),
    }
    append_line(run_info_log, json.dumps(run_metadata, ensure_ascii=False))

    if pending.empty:
        print("没有待处理样本。现有RAG预测已覆盖输入清单。")
        return 0

    print("本次运行ID：{}".format(run_id))
    print("待处理样本：{}".format(len(pending)))
    print("模型：{}".format(model))
    print("检索方法：{}".format(retrieval_method))
    print("Top-K：{}".format(top_k))
    print("真值参与推理：否")

    success_count = 0
    format_error_count = 0
    api_error_count = 0
    input_error_count = 0

    total_pending = len(pending)
    for position, manifest_row in enumerate(
        pending.to_dict(orient="records"), start=1
    ):
        sample_id = str(manifest_row["sample_id"])
        image_path = Path(manifest_row["crop_path"])
        print(
            "[{}/{}] {}".format(position, total_pending, sample_id),
            end=" ",
            flush=True,
        )

        row = blank_prediction_row(run_id, sample_id, model)
        row["retrieval_method"] = retrieval_method

        direct_row = direct_by_id.get(sample_id)
        if direct_row is None:
            row.update(
                {
                    "status": "input_error",
                    "decision": "api_error",
                    "parse_error": "缺少直接基线预测结果",
                }
            )
            upsert_prediction(prediction_rows, row)
            write_predictions_atomic(predictions_path, prediction_rows)
            input_error_count += 1
            print("输入错误")
            continue

        direct_payload = direct_visual_payload(direct_row)
        retrieval_payload, filter_audit = sanitize_direct_payload_for_retrieval(
            direct_payload
        )
        row["direct_status"] = clean_cell(direct_payload.get("status"))
        row["direct_decision"] = clean_cell(direct_payload.get("decision"))
        row["original_direct_visual_result_json"] = json.dumps(
            direct_payload, ensure_ascii=False
        )
        row["sanitized_direct_visual_result_json"] = json.dumps(
            retrieval_payload, ensure_ascii=False
        )
        row["retrieval_filter_applied"] = (
            "1" if filter_audit.get("filter_applied") else "0"
        )
        row["removed_visible_text_json"] = json.dumps(
            filter_audit.get("removed_visible_text", []), ensure_ascii=False
        )
        row["retrieval_filter_reasons_json"] = json.dumps(
            filter_audit.get("filter_reasons", []), ensure_ascii=False
        )

        if not image_path.exists() or not image_path.is_file():
            row.update(
                {
                    "status": "input_error",
                    "decision": "api_error",
                    "parse_error": "图片不存在：{}".format(image_path),
                }
            )
            upsert_prediction(prediction_rows, row)
            write_predictions_atomic(predictions_path, prediction_rows)
            append_line(
                api_errors_log,
                "{}\t{}\tinput_error\t{}".format(now_iso(), sample_id, image_path),
            )
            input_error_count += 1
            print("输入错误")
            continue

        query_text = build_retrieval_query(retrieval_payload)
        candidates = retrieve_candidates(
            query_text=query_text,
            payload=retrieval_payload,
            bundle=bundle,
            top_k=top_k,
            boosts=field_boosts,
        )
        compact_candidates = candidate_prompt_payload(candidates)

        row["retrieval_query"] = query_text
        row["retrieved_candidate_count"] = len(candidates)
        row["retrieved_candidates_json"] = json.dumps(
            candidates, ensure_ascii=False
        )
        if candidates:
            row["retrieval_top1_sku_id"] = candidates[0]["sku_id"]
            row["retrieval_top1_score"] = candidates[0]["retrieval_score"]

        retrieval_record = {
            "run_id": run_id,
            "sample_id": sample_id,
            "processed_at": now_iso(),
            "query_source": retrieval_cfg.get("query_source"),
            "retrieval_method": retrieval_method,
            "query_text": query_text,
            "original_direct_visual_result": direct_payload,
            "sanitized_direct_visual_result": retrieval_payload,
            "retrieval_filter_audit": filter_audit,
            "candidates": candidates,
            "ground_truth_used": False,
        }
        append_jsonl(retrieval_results_path, retrieval_record)

        try:
            data_url = image_to_data_url(image_path)
        except Exception as exc:
            row.update(
                {
                    "status": "input_error",
                    "decision": "api_error",
                    "parse_error": str(exc),
                }
            )
            upsert_prediction(prediction_rows, row)
            write_predictions_atomic(predictions_path, prediction_rows)
            append_line(
                api_errors_log,
                "{}\t{}\tinput_error\t{}".format(now_iso(), sample_id, exc),
            )
            input_error_count += 1
            print("输入错误")
            continue

        user_prompt = user_prompt_template.format(
            sample_id=sample_id,
            top_k=top_k,
            direct_visual_result=json.dumps(
                retrieval_payload, ensure_ascii=False, indent=2
            ),
            candidate_context=json.dumps(
                compact_candidates, ensure_ascii=False, indent=2
            ),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        completion = None
        latency_seconds = None
        attempt_used = 0
        last_error = None

        for attempt in range(1, max_retries + 1):
            attempt_used = attempt
            started = time.perf_counter()
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    extra_body={
                        "enable_thinking": enable_thinking,
                        "vl_high_resolution_images": vl_high_resolution_images,
                    },
                    stream=False,
                )
                latency_seconds = round(time.perf_counter() - started, 4)
                last_error = None
                break
            except Exception as exc:
                latency_seconds = round(time.perf_counter() - started, 4)
                last_error = exc
                append_line(
                    api_errors_log,
                    "{}\t{}\tattempt={}/{}\t{}: {}".format(
                        now_iso(),
                        sample_id,
                        attempt,
                        max_retries,
                        type(exc).__name__,
                        exc,
                    ),
                )
                if attempt < max_retries:
                    time.sleep(get_retry_delay(attempt))

        if completion is None:
            row.update(
                {
                    "status": "api_error",
                    "decision": "api_error",
                    "latency_seconds": latency_seconds,
                    "attempt_count": attempt_used,
                    "parse_error": "{}: {}".format(
                        type(last_error).__name__ if last_error else "APIError",
                        last_error or "未知API错误",
                    ),
                }
            )
            upsert_prediction(prediction_rows, row)
            write_predictions_atomic(predictions_path, prediction_rows)
            api_error_count += 1
            print("API失败")
            continue

        message = completion.choices[0].message
        content = message.content or ""
        finish_reason = getattr(completion.choices[0], "finish_reason", None)
        response_id = getattr(completion, "id", None)
        response_model = getattr(completion, "model", None) or model
        prompt_tokens, completion_tokens, total_tokens, reasoning_tokens = usage_values(
            completion
        )

        raw_record = {
            "run_id": run_id,
            "sample_id": sample_id,
            "processed_at": now_iso(),
            "model_requested": model,
            "model_returned": response_model,
            "response_id": response_id,
            "finish_reason": finish_reason,
            "content": content,
            "original_direct_visual_result": direct_payload,
            "sanitized_direct_visual_result": retrieval_payload,
            "retrieval_filter_audit": filter_audit,
            "retrieval_query": query_text,
            "retrieved_candidates": candidates,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
            "latency_seconds": latency_seconds,
            "attempt_count": attempt_used,
            "ground_truth_used": False,
            "reasoning_content_saved": False,
        }
        append_jsonl(raw_responses_path, raw_record)

        row.update(
            {
                "model": str(response_model),
                "response_id": response_id or "",
                "finish_reason": finish_reason or "",
                "prompt_tokens": prompt_tokens if prompt_tokens is not None else "",
                "completion_tokens": completion_tokens
                if completion_tokens is not None
                else "",
                "total_tokens": total_tokens if total_tokens is not None else "",
                "reasoning_tokens": reasoning_tokens
                if reasoning_tokens is not None
                else "",
                "latency_seconds": latency_seconds,
                "attempt_count": attempt_used,
            }
        )

        try:
            parsed = parse_json_answer(content)
            validated = validate_prediction(parsed, sample_id, candidates)

            # 先保存模型原始字段，再进行候选SKU的确定性回填。
            row["model_pred_drug_name"] = validated["pred_drug_name"] or ""
            row["model_pred_generic_name"] = validated["pred_generic_name"] or ""
            row["model_pred_specification"] = (
                validated["pred_specification"] or ""
            )
            row["model_pred_manufacturer"] = validated["pred_manufacturer"] or ""
            row["model_pred_barcode_or_approval_no"] = (
                validated["pred_barcode_or_approval_no"] or ""
            )
            row["model_pred_sku_id"] = validated["pred_sku_id"] or ""

            guarded_result, guard_audit = apply_non_product_context_guard(
                validated, filter_audit
            )
            final_result = canonicalize_selected_candidate(
                guarded_result, candidates
            )
            row["context_guard_triggered"] = (
                "1" if guard_audit.get("triggered") else "0"
            )
            row["context_guard_reason"] = json.dumps(
                guard_audit.get("reasons", []), ensure_ascii=False
            )
            row.update(
                {
                    "status": "ok",
                    "decision": final_result["decision"],
                    "pred_drug_name": final_result["pred_drug_name"] or "",
                    "pred_generic_name": final_result["pred_generic_name"] or "",
                    "pred_specification": final_result["pred_specification"] or "",
                    "pred_manufacturer": final_result["pred_manufacturer"] or "",
                    "pred_barcode_or_approval_no": final_result[
                        "pred_barcode_or_approval_no"
                    ]
                    or "",
                    "pred_sku_id": final_result["pred_sku_id"] or "",
                    "confidence": ""
                    if final_result["confidence"] is None
                    else final_result["confidence"],
                    "visible_text_json": json.dumps(
                        final_result["visible_text"], ensure_ascii=False
                    ),
                    "package_features_json": json.dumps(
                        final_result["package_features"], ensure_ascii=False
                    ),
                    "selected_candidate_rank": final_result[
                        "selected_candidate_rank"
                    ]
                    or "",
                    "candidate_consistency": final_result[
                        "candidate_consistency"
                    ],
                    "evidence_from_image_json": json.dumps(
                        final_result["evidence_from_image"], ensure_ascii=False
                    ),
                    "evidence_from_kb_json": json.dumps(
                        final_result["evidence_from_kb"], ensure_ascii=False
                    ),
                    "reason": final_result["reason"] or "",
                    "parse_error": "",
                }
            )
            success_count += 1
            selected_text = row["pred_sku_id"] or "无SKU"
            print("完成：{}，{}".format(final_result["decision"], selected_text))
        except Exception as exc:
            error_message = "{}: {}".format(type(exc).__name__, exc)
            row.update(
                {
                    "status": "format_error",
                    "decision": "format_error",
                    "parse_error": error_message,
                }
            )
            format_error_count += 1
            print("格式错误：{}".format(error_message))

        upsert_prediction(prediction_rows, row)
        write_predictions_atomic(predictions_path, prediction_rows)

    finish_metadata = {
        "run_id": run_id,
        "finished_at": now_iso(),
        "success_count": success_count,
        "format_error_count": format_error_count,
        "api_error_count": api_error_count,
        "input_error_count": input_error_count,
        "predictions_path": str(predictions_path),
        "retrieval_results_path": str(retrieval_results_path),
        "raw_responses_path": str(raw_responses_path),
    }
    append_line(run_info_log, json.dumps(finish_metadata, ensure_ascii=False))

    print("\n本次运行结束。")
    print("成功：{}".format(success_count))
    print("格式错误：{}".format(format_error_count))
    print("API失败：{}".format(api_error_count))
    print("输入错误：{}".format(input_error_count))
    print("预测表：{}".format(predictions_path))
    print("检索记录：{}".format(retrieval_results_path))
    print("\n确认没有api_error后，运行：python 04_evaluate_rag_results.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断。已完成样本已安全保存。", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print("\n运行失败：{}".format(error), file=sys.stderr)
        sys.exit(1)
