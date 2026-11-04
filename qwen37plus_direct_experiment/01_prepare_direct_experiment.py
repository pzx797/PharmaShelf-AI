# -*- coding: utf-8 -*-
"""Prepare the Qwen3.7-Plus direct drug-crop recognition experiment.

This script:
1. Validates the crop directory, ground-truth workbook, and knowledge base.
2. Recursively scans crop images.
3. Matches each image to the ground-truth row through the ``crop_path`` column.
4. Generates stable anonymous sample IDs and ``input_manifest.csv``.
5. Creates the experiment directory, ``config.yaml``, and frozen prompt files.

Run:
    python 01_prepare_direct_experiment.py

Dependencies:
    pip install pandas openpyxl pyyaml

Python compatibility: Python 3.8+
"""

from __future__ import print_function

import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
    import yaml
except ImportError as exc:
    raise SystemExit(
        "缺少依赖。请先执行：pip install pandas openpyxl pyyaml\n"
        "原始错误：{}".format(exc)
    )


# -----------------------------------------------------------------------------
# Fixed experiment settings
# -----------------------------------------------------------------------------
CROP_DIR = Path(
    r"E:\EdgeDownload\ultralytics-main\runs\MSA_SKU_to_MRS_stage2_seed3_best_crops\crops"
)
GT_PATH = Path(
    r"E:\EdgeDownload\ultralytics-main\runs\MSA_SKU_to_MRS_stage2_seed3_best_crops\instances_gt_v1_enriched.xlsx"
)
OUTPUT_ROOT = Path(
    r"E:\EdgeDownload\ultralytics-main\qwen37plus_direct_experiment"
)
KB_PATH = Path(r"E:\Desktop\Knowledge_Base\sku_kb_cleaned.xlsx")
GT_IMAGE_COLUMN = "crop_path"

RUN_NAME = "qwen37plus_direct_v1"
RUN_DIR = OUTPUT_ROOT / "runs" / RUN_NAME

MODEL_NAME = "qwen3.7-plus-2026-05-26"
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV = "DASHSCOPE_API_KEY"

ENABLE_THINKING = True
TEMPERATURE = 0.1
MAX_TOKENS = 1024
TIMEOUT_SECONDS = 180
MAX_RETRIES = 3
CONCURRENCY = 1
SCAN_SUBDIRECTORIES = True
EVALUATE_ALL_SAMPLES = True
STRICT_SKU_MODE = "B"

# Fine text and small package details are central to this experiment.
# Keep this value unchanged across all comparison groups.
VL_HIGH_RESOLUTION_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


SYSTEM_PROMPT = """你是一名药店货架商品包装识别助手。识别对象为药店货架中的所有待识别商品，不限制商品类别、药品剂型、包装形态或容器形式。

请仅依据当前裁剪图中实际可见的文字、包装颜色、图案、版式和外观信息识别商品，不得使用外部知识库、货架位置、文件名或其他样本信息。

必须遵守以下规则：

1.只要能够从图片中可靠确认商品名称，decision就输出`matched`。
2.商品名称能够确认，但规格、生产厂家、批准文号或条码无法确认时，对应字段填写null，不得因此输出`unknown`。
3.不得仅因商品不是标准盒装药品、没有显示批准文号或采用瓶装、袋装、罐装等包装形式而拒绝识别。
4.只有商品名称本身无法可靠确认时，decision才输出`unknown`。仅能看出品牌、商品类别、部分不完整文字或包装特征时，不能据此猜测完整商品名称。
5.如果图片中包含多个不同商品，且无法确定主要识别对象，decision输出`multi_object`。
6.只有裁剪图中不存在有效药店货架商品，例如仅包含货架、价格标签、背景、手部或其他非商品区域时，decision才输出`non_drug`。
7.不得根据常识补全图片中模糊、遮挡或不可见的文字。
8.pred_drug_name填写包装上能够可靠确认的商品名称；如果包装同时显示规范通用名称，则填写pred_generic_name。
9.pred_sku_id固定填写null。本实验不向模型提供知识库，严格SKU由评价脚本离线映射。
10.visible_text只能记录图片中实际可见的文字，不得写入推测内容。
11.只输出一个合法JSON对象，不得输出Markdown代码块或JSON之外的说明。
12.品牌名称、生产企业名称、系列名称、宣传用语、产地、货架标签或包装上的普通文字，不能单独作为商品名称。只能确认这些信息而无法确认实际商品名称时，decision必须输出unknown。
13.不得仅根据包装中的内容物外观、颜色、形状或药店货架类别推断具体商品名称。没有明确商品名称文字或其他充分身份依据时，必须输出unknown。

JSON字段必须完整：

{
"sample_id": "样本编号",
"decision": "matched|unknown|multi_object|non_drug",
"pred_drug_name": "商品名称或null",
"pred_generic_name": "通用名称或null",
"pred_specification": "规格或null",
"pred_manufacturer": "生产厂家或null",
"pred_barcode_or_approval_no": "批准文号或条码或null",
"pred_sku_id": null,
"confidence": 0.00,
"visible_text": ["图片中实际可见的文字"],
"package_features": ["包装颜色、图案、版式或容器特征"],
"reason": "一句话说明判断依据或无法确认的原因"
}
"""


USER_PROMPT = """请识别当前药店货架商品裁剪图中的商品信息。

样本编号：{sample_id}

请观察并提取：

* 商品名称；
* 通用名称；
* 规格；
* 生产厂家；
* 批准文号或条码；
* 图片中实际可见的文字；
* 包装颜色、图案、版式和容器形式。

只要能够可靠确认商品名称，就输出`matched`。其他无法确认的字段填写null，不得因为规格、厂家或批准信息不完整而输出`unknown`。

药店货架商品可能具有不同类别、剂型和包装形式，不得仅根据包装类型判断其不是有效识别对象。

只有商品名称本身无法确认时才输出`unknown`，不要根据品牌、商品类别或模糊文字猜测完整名称。只输出规定的JSON对象。
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path_text(value: object) -> str:
    """Normalize a path-like cell for matching without requiring it to exist."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    text = str(value).strip().strip('"').strip("'")
    if not text:
        return ""
    text = text.replace("/", "\\")
    text = re.sub(r"\\+", r"\\", text)
    return text.casefold()


def path_match_keys(path_text: object, crop_dir: Path) -> List[str]:
    """Return increasingly broad keys for matching a GT path to an image."""
    normalized = normalize_path_text(path_text)
    if not normalized:
        return []

    keys: List[str] = []

    def add(value: str) -> None:
        value = value.strip().casefold()
        if value and value not in keys:
            keys.append(value)

    add(normalized)
    add(Path(normalized).name)

    crop_dir_norm = normalize_path_text(str(crop_dir))
    if crop_dir_norm and normalized.startswith(crop_dir_norm + "\\"):
        add(normalized[len(crop_dir_norm) + 1 :])

    # Some tables store paths beginning with "crops\\...".
    marker = "\\crops\\"
    if marker in normalized:
        add(normalized.split(marker, 1)[1])

    return keys


def discover_images(root: Path, recursive: bool) -> List[Path]:
    iterator: Iterable[Path]
    iterator = root.rglob("*") if recursive else root.glob("*")
    images = [
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda p: str(p).casefold())


def build_gt_lookup(
    gt_df: pd.DataFrame, image_column: str, crop_dir: Path
) -> Tuple[Dict[str, List[int]], Dict[int, str]]:
    lookup: Dict[str, List[int]] = {}
    normalized_values: Dict[int, str] = {}

    for idx, value in gt_df[image_column].items():
        normalized_values[int(idx)] = normalize_path_text(value)
        for key in path_match_keys(value, crop_dir):
            lookup.setdefault(key, []).append(int(idx))

    return lookup, normalized_values


def match_gt_row(
    image_path: Path,
    crop_dir: Path,
    lookup: Dict[str, List[int]],
) -> Tuple[Optional[int], str, int]:
    relative_path = image_path.relative_to(crop_dir.resolve())
    candidate_keys = [
        normalize_path_text(str(image_path)),
        normalize_path_text(str(relative_path)),
        image_path.name.casefold(),
    ]

    # Match from the most specific key to the basename.
    for key in candidate_keys:
        rows = sorted(set(lookup.get(key, [])))
        if len(rows) == 1:
            return rows[0], key, 1
        if len(rows) > 1:
            return None, key, len(rows)

    return None, "", 0


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def ensure_required_paths() -> None:
    missing = []
    for label, path in [
        ("裁剪图目录", CROP_DIR),
        ("人工真值表", GT_PATH),
        ("知识库", KB_PATH),
    ]:
        if not path.exists():
            missing.append("{}不存在：{}".format(label, path))
    if missing:
        raise FileNotFoundError("\n".join(missing))


def build_config() -> Dict[str, object]:
    return {
        "experiment": {
            "name": RUN_NAME,
            "version": "1.0",
            "created_at": now_iso(),
            "task": "Qwen3.7-Plus无知识库药品裁剪图直接识别",
            "evaluate_all_samples": EVALUATE_ALL_SAMPLES,
            "strict_sku_mode": STRICT_SKU_MODE,
        },
        "paths": {
            "crop_dir": str(CROP_DIR),
            "ground_truth_path": str(GT_PATH),
            "knowledge_base_path": str(KB_PATH),
            "output_root": str(OUTPUT_ROOT),
            "run_dir": str(RUN_DIR),
            "manifest_path": str(RUN_DIR / "input_manifest.csv"),
            "system_prompt_path": str(RUN_DIR / "prompts" / "system_prompt.txt"),
            "user_prompt_path": str(RUN_DIR / "prompts" / "user_prompt.txt"),
            "raw_responses_path": str(RUN_DIR / "responses" / "raw_responses.jsonl"),
            "predictions_path": str(RUN_DIR / "predictions" / "direct_predictions.csv"),
            "merged_results_path": str(RUN_DIR / "evaluation" / "merged_results.csv"),
            "metrics_path": str(RUN_DIR / "evaluation" / "metrics.json"),
            "error_cases_path": str(RUN_DIR / "evaluation" / "error_cases.csv"),
            "api_errors_log": str(RUN_DIR / "logs" / "api_errors.log"),
            "run_info_log": str(RUN_DIR / "logs" / "run_info.log"),
        },
        "data": {
            "ground_truth_image_column": GT_IMAGE_COLUMN,
            "scan_subdirectories": SCAN_SUBDIRECTORIES,
            "image_extensions": sorted(IMAGE_EXTENSIONS),
        },
        "api": {
            "platform": "阿里云百炼",
            "base_url": API_BASE_URL,
            "api_key_env": API_KEY_ENV,
            "model": MODEL_NAME,
            "enable_thinking": ENABLE_THINKING,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_retries": MAX_RETRIES,
            "concurrency": CONCURRENCY,
            "response_format": "json_object",
            "vl_high_resolution_images": VL_HIGH_RESOLUTION_IMAGES,
        },
        "evaluation": {
            "column_candidates": {
                "gt_sku_id": ["gt_sku_id"],
                "gt_drug_name": ["gt_drug_name", "gt_generic_name"],
                "gt_specification": ["gt_spec", "gt_specification"],
                "gt_manufacturer": ["gt_manufacturer"],
                "gt_barcode_or_approval_no": ["gt_barcode_or_approval_no"],
                "label_status": ["label_status"],
                "visibility": ["visibility"],
                "readable": ["readable"],
                "multi_object": ["multi_object"],
                "false_positive": ["false_positive"],
                "note": ["note"],
            },
            "knowledge_base_columns": {
                "sku_id": ["sku_id"],
                "drug_name": ["drug_name", "generic_name", "raw_product_name"],
                "specification": ["specification", "raw_specification"],
                "manufacturer": ["manufacturer"],
                "aliases": ["aliases"],
            },
        },
    }


def main() -> int:
    print("正在准备Qwen3.7-Plus直接识别实验……")
    ensure_required_paths()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ["prompts", "responses", "predictions", "evaluation", "logs"]:
        (RUN_DIR / subdir).mkdir(parents=True, exist_ok=True)

    try:
        gt_df = pd.read_excel(GT_PATH, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError("读取人工真值表失败：{}".format(exc))

    if GT_IMAGE_COLUMN not in gt_df.columns:
        raise KeyError(
            "真值表中不存在图片关联列“{}”。当前表头：{}".format(
                GT_IMAGE_COLUMN, "、".join(map(str, gt_df.columns.tolist()))
            )
        )

    images = discover_images(CROP_DIR, SCAN_SUBDIRECTORIES)
    if not images:
        raise RuntimeError("裁剪图目录中未找到支持的图片：{}".format(CROP_DIR))

    gt_lookup, _ = build_gt_lookup(gt_df, GT_IMAGE_COLUMN, CROP_DIR)

    rows: List[Dict[str, object]] = []
    matched_count = 0
    ambiguous_count = 0
    unmatched_count = 0

    for number, image_path in enumerate(images, start=1):
        relative_path = image_path.relative_to(CROP_DIR.resolve())
        gt_index, match_key, candidate_count = match_gt_row(
            image_path, CROP_DIR, gt_lookup
        )
        if gt_index is not None:
            gt_match_status = "matched"
            matched_count += 1
            gt_excel_row = gt_index + 2  # pandas index 0 -> Excel row 2
        elif candidate_count > 1:
            gt_match_status = "ambiguous"
            ambiguous_count += 1
            gt_excel_row = ""
        else:
            gt_match_status = "unmatched"
            unmatched_count += 1
            gt_excel_row = ""

        rows.append(
            {
                "sample_id": "sample_{:06d}".format(number),
                "crop_path": str(image_path),
                "crop_relative_path": str(relative_path),
                "crop_filename": image_path.name,
                "image_extension": image_path.suffix.lower(),
                "image_size_bytes": image_path.stat().st_size,
                "image_sha256": sha256_file(image_path),
                "gt_match_status": gt_match_status,
                "gt_row_index": "" if gt_index is None else gt_index,
                "gt_excel_row": gt_excel_row,
                "gt_match_key": match_key,
                "include_in_inference": 1,
            }
        )

    manifest_df = pd.DataFrame(rows)
    manifest_path = RUN_DIR / "input_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    config = build_config()
    config_path = RUN_DIR / "config.yaml"
    with config_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(
            config,
            file_obj,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    write_text(RUN_DIR / "prompts" / "system_prompt.txt", SYSTEM_PROMPT)
    write_text(RUN_DIR / "prompts" / "user_prompt.txt", USER_PROMPT)

    summary = {
        "prepared_at": now_iso(),
        "run_dir": str(RUN_DIR),
        "image_count": len(images),
        "ground_truth_row_count": len(gt_df),
        "matched_image_count": matched_count,
        "ambiguous_image_count": ambiguous_count,
        "unmatched_image_count": unmatched_count,
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
    }
    write_text(
        RUN_DIR / "logs" / "prepare_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )

    print("\n准备完成。")
    print("实验目录：{}".format(RUN_DIR))
    print("裁剪图数量：{}".format(len(images)))
    print("成功匹配真值：{}".format(matched_count))
    print("匹配歧义：{}".format(ambiguous_count))
    print("未匹配真值：{}".format(unmatched_count))
    print("配置文件：{}".format(config_path))
    print("输入清单：{}".format(manifest_path))

    if ambiguous_count or unmatched_count:
        print(
            "\n注意：存在未唯一匹配真值的图片。API推理仍可执行，但评价脚本会将这些样本标记为不可评价。"
        )

    print("\n下一步运行：python 02_run_qwen37plus_direct.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("\n准备失败：{}".format(error), file=sys.stderr)
        sys.exit(1)
