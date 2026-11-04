# -*- coding: utf-8 -*-
"""准备Qwen3.7-Plus药品裁剪图RAG识别实验。

实验设计：
1.复用已完成直接识别基线的input_manifest.csv和direct_predictions.csv。
2.第一阶段视觉结果被冻结，仅作为知识库检索查询，不重新调用API。
3.RAG推理阶段只增加知识库候选检索与候选约束复核，不读取人工真值字段。
4.生成独立的实验目录、配置文件、提示词和输入清单。

运行：
    python 01_prepare_rag_experiment.py

依赖：
    pip install pandas openpyxl pyyaml

兼容：Python 3.8+
"""

from __future__ import print_function

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd
    import yaml
except ImportError as exc:
    raise SystemExit(
        "缺少依赖。请先执行：pip install pandas openpyxl pyyaml\n"
        "原始错误：{}".format(exc)
    )


# -----------------------------------------------------------------------------
# 固定实验路径：均按用户当前Windows目录设置
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(r"E:\EdgeDownload\ultralytics-main")

CROP_DIR = PROJECT_ROOT / r"runs\MSA_SKU_to_MRS_stage2_seed3_best_crops\crops"
GT_PATH = (
    PROJECT_ROOT
    / r"runs\MSA_SKU_to_MRS_stage2_seed3_best_crops\instances_gt_v1_enriched.xlsx"
)
KB_PATH = PROJECT_ROOT / r"rag_knowledge_base\sku_kb_cleaned.xlsx"

DIRECT_EXPERIMENT_ROOT = PROJECT_ROOT / "qwen37plus_direct_experiment"
DIRECT_RUN_NAME = "qwen37plus_direct_v1"
DIRECT_RUN_DIR = DIRECT_EXPERIMENT_ROOT / "runs" / DIRECT_RUN_NAME
DIRECT_CONFIG_PATH = DIRECT_RUN_DIR / "config.yaml"
DIRECT_MANIFEST_PATH = DIRECT_RUN_DIR / "input_manifest.csv"
DIRECT_PREDICTIONS_PATH = (
    DIRECT_RUN_DIR / "predictions" / "direct_predictions.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "qwen37plus_rag_experiment"
RUN_NAME = "qwen37plus_rag_v1"
RUN_DIR = OUTPUT_ROOT / "runs" / RUN_NAME

MODEL_NAME = "qwen3.7-plus-2026-05-26"
API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV = "DASHSCOPE_API_KEY"

ENABLE_THINKING = True
TEMPERATURE = 0.1
MAX_TOKENS = 1024
TIMEOUT_SECONDS = 300
MAX_RETRIES = 3
CONCURRENCY = 1
VL_HIGH_RESOLUTION_IMAGES = True

# RAG检索设置：采用本地TF-IDF字符级稀疏检索，并加入字段精确匹配加权。
TOP_K = 5
TFIDF_NGRAM_MIN = 2
TFIDF_NGRAM_MAX = 4
TFIDF_MAX_FEATURES = 120000
RETRIEVAL_METHOD = "char_tfidf_field_boost_v1"
QUERY_SOURCE = "frozen_direct_predictions"

# 与直接基线保持一致，用于评价时排除提示词调试样本。
PROMPT_DEVELOPMENT_SAMPLE_COUNT = 10
STRICT_SKU_MODE = "B"


SYSTEM_PROMPT = """你是一名药店货架商品包装识别助手。当前任务是在裁剪商品图和知识库候选记录的共同约束下，识别商品名称并尽可能确定唯一SKU。

知识库候选由检索程序根据第一阶段视觉线索自动召回，只是候选集合，不是真值。你必须重新观察当前裁剪图，判断候选是否得到图像证据支持。

必须遵守以下规则：

1.只能从提供的候选集合中选择pred_sku_id，不得生成候选集合之外的SKU编号。
2.候选排名不能单独作为选择依据。必须依据图中实际可见文字、包装颜色、图案、版式、规格和厂家等证据判断。
3.当图像证据足以确认商品名称，并且某个候选与图像证据一致时，decision输出`matched`，pred_sku_id填写该候选SKU。
4.当商品名称能够从图中可靠确认，但没有候选与其一致时，decision仍可输出`matched`，pred_sku_id填写null，其他无法确认字段填写null。
5.当商品名称本身无法可靠确认，或多个候选无法根据图像证据排除时，decision输出`unknown`，不得仅根据检索第一名猜测。
6.如果图中包含多个不同商品且无法确定主要识别对象，decision输出`multi_object`。
7.只有图中不存在有效药店货架商品时，decision才输出`non_drug`。
8.知识库字段可用于补全已经由图像证据确定的同一候选记录，但不得把候选中的文字错误描述为图中可见文字。
9.visible_text只能记录图片中实际可见的文字；evidence_from_kb只能记录知识库候选提供的信息。
10.如果选择候选，pred_drug_name、pred_generic_name、pred_specification、pred_manufacturer和pred_barcode_or_approval_no应与该候选记录保持一致。
11.只输出一个合法JSON对象，不得输出Markdown代码块或JSON之外的说明。
12.选择候选SKU时，selected_candidate_rank必须填写该候选的实际排名，candidate_consistency必须填写strong或partial；未选择候选SKU时，selected_candidate_rank必须填写null，candidate_consistency必须填写none，两个字段均不得留空。
13.品牌名称、商标、系列名称、生产企业名称、宣传用语、产地、门店名称、货架标签和价格标签不能单独作为商品名称。只有这些信息可见而商品包装上的实际商品名称无法确认时，decision必须输出unknown。
14.不得根据瓶内或包装内物品的颜色、形状、颗粒、切片、粉末等外观特征推断具体商品名称。内容物外观只能作为辅助包装特征，不能作为确定商品身份的主要依据。
15.必须区分品牌、商标、门店名称和生产厂家。只有图片中明确出现“生产企业”“生产厂家”“制造商”“上市许可持有人”等字段时，才能把对应文字认定为生产厂家。“中国药材”“健之佳”“向辉”等文字可能是品牌、系列或门店标识，不得仅凭其出现就判定为生产厂家。
16.候选判断的证据优先级依次为：条码或批准文号完全一致、商品名称和规格一致、商品名称和明确标注的生产厂家一致、商品名称和包装特征一致。当商品名称和规格与候选一致，且图中没有明确标注的厂家或条码冲突时，可以选择该候选；不得仅因品牌或商标与候选厂家名称不同而拒绝候选。
17.货架标签和价格标签只能作为场景信息，不能用于确认裁剪商品的名称、规格、厂家或SKU，也不得写入支持商品身份的evidence_from_image。

JSON字段必须完整：

{
"sample_id": "样本编号",
"decision": "matched|unknown|multi_object|non_drug",
"pred_drug_name": "商品名称或null",
"pred_generic_name": "通用名称或null",
"pred_specification": "规格或null",
"pred_manufacturer": "生产厂家或null",
"pred_barcode_or_approval_no": "批准文号或条码或null",
"pred_sku_id": "候选SKU编号或null",
"selected_candidate_rank": "所选候选的实际排名或null",
"candidate_consistency": "strong|partial|none；未选择SKU时必须为none",
"confidence": 0.00,
"visible_text": ["图片中实际可见的文字"],
"package_features": ["包装颜色、图案、版式或容器特征"],
"evidence_from_image": ["支持或否定候选的图像证据"],
"evidence_from_kb": ["所选候选的知识库证据"],
"reason": "一句话说明选择候选或拒绝判断的原因"
}
"""


USER_PROMPT = """请识别当前药店货架商品裁剪图，并结合给出的知识库候选确定最终结果。

样本编号：{sample_id}

第一阶段视觉识别结果：
{direct_visual_result}

知识库Top-{top_k}候选：
{candidate_context}

请重新观察图片并完成以下判断：

* 检查第一阶段视觉线索是否与图片一致；
* 比较各候选的商品名称、规格、条码、明确标注的生产厂家和包装关键词；
* 不得把品牌、商标、门店名称、货架标签或价格标签当作生产厂家或商品名称证据；
* 当商品名称和规格与候选一致，且不存在明确的条码或生产厂家冲突时，可以选择该候选；
* 只有图像证据支持时才能选择候选SKU；
* 当商品名称本身无法可靠确认，或多个候选无法排除时，输出unknown；
* 当商品名称可以从图中可靠确认，但没有候选与其一致时，输出matched，pred_sku_id填写null；
* 选择候选时填写实际selected_candidate_rank，并将candidate_consistency填写为strong或partial；未选择候选时分别填写null和none；
* 不得把知识库候选文字写入visible_text。

只输出规定的JSON对象。
"""


# -----------------------------------------------------------------------------
# 通用辅助函数
# -----------------------------------------------------------------------------
def now_iso() -> str:
    """返回带本地时区的ISO时间字符串。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """计算文件SHA256，用于冻结并核验实验输入。"""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, content: str) -> None:
    """以UTF-8写入文本，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def read_yaml(path: Path) -> Dict[str, Any]:
    """读取YAML配置文件并检查顶层结构。"""
    with path.open("r", encoding="utf-8") as file_obj:
        value = yaml.safe_load(file_obj)
    if not isinstance(value, dict):
        raise ValueError("YAML配置内容无效：{}".format(path))
    return value


def ensure_required_paths() -> None:
    """检查RAG实验依赖的目录和文件是否齐全。"""
    required = [
        ("裁剪图目录", CROP_DIR),
        ("人工真值表", GT_PATH),
        ("知识库", KB_PATH),
        ("直接基线配置", DIRECT_CONFIG_PATH),
        ("直接基线输入清单", DIRECT_MANIFEST_PATH),
        ("直接基线预测结果", DIRECT_PREDICTIONS_PATH),
    ]
    missing: List[str] = []
    for label, path in required:
        if not path.exists():
            missing.append("{}不存在：{}".format(label, path))
    if missing:
        raise FileNotFoundError("\n".join(missing))


def validate_direct_artifacts(
    manifest: pd.DataFrame,
    direct_predictions: pd.DataFrame,
) -> Dict[str, int]:
    """验证直接基线清单和预测结果能够按sample_id一一对应。"""
    manifest_required = {
        "sample_id",
        "crop_path",
        "crop_filename",
        "image_sha256",
        "gt_match_status",
        "gt_row_index",
        "include_in_inference",
    }
    prediction_required = {
        "sample_id",
        "status",
        "decision",
        "pred_drug_name",
        "pred_generic_name",
        "pred_specification",
        "pred_manufacturer",
        "pred_barcode_or_approval_no",
        "visible_text_json",
        "package_features_json",
        "reason",
    }

    missing_manifest = manifest_required - set(manifest.columns)
    missing_predictions = prediction_required - set(direct_predictions.columns)
    if missing_manifest:
        raise KeyError(
            "直接基线input_manifest.csv缺少字段：{}".format(
                sorted(missing_manifest)
            )
        )
    if missing_predictions:
        raise KeyError(
            "直接基线direct_predictions.csv缺少字段：{}".format(
                sorted(missing_predictions)
            )
        )

    if manifest["sample_id"].duplicated().any():
        raise ValueError("直接基线输入清单存在重复sample_id")
    if direct_predictions["sample_id"].duplicated().any():
        raise ValueError("直接基线预测结果存在重复sample_id")

    manifest_ids = set(manifest["sample_id"].astype(str))
    prediction_ids = set(direct_predictions["sample_id"].astype(str))
    missing_predictions_ids = sorted(manifest_ids - prediction_ids)
    extra_prediction_ids = sorted(prediction_ids - manifest_ids)

    if missing_predictions_ids:
        raise ValueError(
            "直接基线仍有{}个样本缺少预测，例如：{}".format(
                len(missing_predictions_ids), missing_predictions_ids[:5]
            )
        )
    if extra_prediction_ids:
        raise ValueError(
            "直接预测结果存在{}个清单外样本，例如：{}".format(
                len(extra_prediction_ids), extra_prediction_ids[:5]
            )
        )

    return {
        "manifest_count": int(len(manifest)),
        "prediction_count": int(len(direct_predictions)),
        "direct_ok_count": int((direct_predictions["status"] == "ok").sum()),
        "direct_non_ok_count": int((direct_predictions["status"] != "ok").sum()),
    }


def build_config(direct_config: Dict[str, Any]) -> Dict[str, Any]:
    """构建RAG实验的冻结配置，保持API参数与直接基线一致。"""
    baseline_api = direct_config.get("api", {})

    # 优先继承已运行基线中的API参数，避免对照实验产生额外变量。
    api_config = {
        "platform": baseline_api.get("platform", "阿里云百炼"),
        "base_url": baseline_api.get("base_url", API_BASE_URL),
        "api_key_env": baseline_api.get("api_key_env", API_KEY_ENV),
        "model": baseline_api.get("model", MODEL_NAME),
        "enable_thinking": baseline_api.get("enable_thinking", ENABLE_THINKING),
        "temperature": baseline_api.get("temperature", TEMPERATURE),
        "max_tokens": baseline_api.get("max_tokens", MAX_TOKENS),
        "timeout_seconds": baseline_api.get("timeout_seconds", TIMEOUT_SECONDS),
        "max_retries": baseline_api.get("max_retries", MAX_RETRIES),
        "concurrency": baseline_api.get("concurrency", CONCURRENCY),
        "response_format": "json_object",
        "vl_high_resolution_images": baseline_api.get(
            "vl_high_resolution_images", VL_HIGH_RESOLUTION_IMAGES
        ),
    }

    return {
        "experiment": {
            "name": RUN_NAME,
            "version": "1.0",
            "created_at": now_iso(),
            "task": "Qwen3.7-Plus药品裁剪图知识库增强识别",
            "comparison_baseline": DIRECT_RUN_NAME,
            "query_source": QUERY_SOURCE,
            "evaluate_all_samples": True,
            "strict_sku_mode": STRICT_SKU_MODE,
            "prompt_development_sample_count": PROMPT_DEVELOPMENT_SAMPLE_COUNT,
            "ground_truth_used_during_inference": False,
        },
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "crop_dir": str(CROP_DIR),
            "ground_truth_path": str(GT_PATH),
            "knowledge_base_path": str(KB_PATH),
            "direct_config_path": str(DIRECT_CONFIG_PATH),
            "direct_manifest_path": str(DIRECT_MANIFEST_PATH),
            "direct_predictions_path": str(DIRECT_PREDICTIONS_PATH),
            "output_root": str(OUTPUT_ROOT),
            "run_dir": str(RUN_DIR),
            "manifest_path": str(RUN_DIR / "input_manifest.csv"),
            "system_prompt_path": str(RUN_DIR / "prompts" / "system_prompt.txt"),
            "user_prompt_path": str(RUN_DIR / "prompts" / "user_prompt.txt"),
            "retrieval_index_path": str(
                RUN_DIR / "index" / "rag_tfidf_index.joblib"
            ),
            "retrieval_records_path": str(
                RUN_DIR / "index" / "knowledge_base_records.csv"
            ),
            "retrieval_metadata_path": str(
                RUN_DIR / "index" / "index_metadata.json"
            ),
            "retrieval_results_path": str(
                RUN_DIR / "retrieval" / "retrieval_results.jsonl"
            ),
            "raw_responses_path": str(
                RUN_DIR / "responses" / "raw_responses.jsonl"
            ),
            "predictions_path": str(
                RUN_DIR / "predictions" / "rag_predictions.csv"
            ),
            "merged_results_path": str(
                RUN_DIR / "evaluation" / "merged_results.csv"
            ),
            "metrics_path": str(RUN_DIR / "evaluation" / "metrics.json"),
            "paper_metrics_path": str(
                RUN_DIR / "evaluation" / "paper_metrics.csv"
            ),
            "error_cases_path": str(
                RUN_DIR / "evaluation" / "error_cases.csv"
            ),
            "manual_review_cases_path": str(
                RUN_DIR / "evaluation" / "manual_review_cases.csv"
            ),
            "retrieval_merged_results_path": str(
                RUN_DIR / "evaluation" / "retrieval_merged_results.csv"
            ),
            "retrieval_metrics_path": str(
                RUN_DIR / "evaluation" / "retrieval_metrics.json"
            ),
            "retrieval_metrics_csv_path": str(
                RUN_DIR / "evaluation" / "retrieval_metrics.csv"
            ),
            "retrieval_error_cases_path": str(
                RUN_DIR / "evaluation" / "retrieval_error_cases.csv"
            ),
            "api_errors_log": str(RUN_DIR / "logs" / "api_errors.log"),
            "run_info_log": str(RUN_DIR / "logs" / "run_info.log"),
        },
        "data": {
            "ground_truth_sheet": "instances_gt",
            "ground_truth_image_column": "crop_path",
            "knowledge_base_sheet": "sku_kb_cleaned",
        },
        "retrieval": {
            "method": RETRIEVAL_METHOD,
            "top_k": TOP_K,
            "query_source": QUERY_SOURCE,
            "tfidf": {
                "analyzer": "char",
                "ngram_range": [TFIDF_NGRAM_MIN, TFIDF_NGRAM_MAX],
                "max_features": TFIDF_MAX_FEATURES,
                "sublinear_tf": True,
                "norm": "l2",
            },
            "field_boosts": {
                "barcode_exact": 1.20,
                "drug_name_exact": 0.85,
                "drug_name_contains": 0.35,
                "generic_name_exact": 0.75,
                "specification_exact": 0.30,
                "manufacturer_exact": 0.20,
                "alias_exact": 0.35,
            },
        },
        "api": api_config,
        "evaluation": {
            "column_candidates": {
                "gt_sku_id": ["gt_sku_id"],
                "gt_drug_name": ["gt_drug_name", "gt_generic_name"],
                "gt_specification": ["gt_spec", "gt_specification"],
                "gt_manufacturer": ["gt_manufacturer"],
                "gt_barcode_or_approval_no": [
                    "gt_barcode_or_approval_no",
                    "gt_barcode",
                ],
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
    """执行目录创建、基线校验、清单复制和配置冻结。"""
    print("正在准备Qwen3.7-Plus RAG识别实验……")
    ensure_required_paths()

    direct_config = read_yaml(DIRECT_CONFIG_PATH)
    manifest = pd.read_csv(
        DIRECT_MANIFEST_PATH,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    direct_predictions = pd.read_csv(
        DIRECT_PREDICTIONS_PATH,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    validation = validate_direct_artifacts(manifest, direct_predictions)

    # 只创建RAG实验所需目录，不改动直接基线文件夹。
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in [
        "prompts",
        "index",
        "retrieval",
        "responses",
        "predictions",
        "evaluation",
        "logs",
    ]:
        (RUN_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # 原样复制直接基线清单，保证两组实验sample_id和图片完全一致。
    rag_manifest_path = RUN_DIR / "input_manifest.csv"
    shutil.copy2(DIRECT_MANIFEST_PATH, rag_manifest_path)

    config = build_config(direct_config)
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
        "baseline_run": DIRECT_RUN_NAME,
        "baseline_config_sha256": sha256_file(DIRECT_CONFIG_PATH),
        "baseline_manifest_sha256": sha256_file(DIRECT_MANIFEST_PATH),
        "baseline_predictions_sha256": sha256_file(DIRECT_PREDICTIONS_PATH),
        "knowledge_base_sha256": sha256_file(KB_PATH),
        "ground_truth_sha256": sha256_file(GT_PATH),
        "query_source": QUERY_SOURCE,
        "ground_truth_used_during_inference": False,
        "validation": validation,
        "config_path": str(config_path),
        "manifest_path": str(rag_manifest_path),
    }
    write_text(
        RUN_DIR / "logs" / "prepare_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )

    print("\n准备完成。")
    print("RAG实验目录：{}".format(RUN_DIR))
    print("输入样本数：{}".format(validation["manifest_count"]))
    print("直接基线成功结果：{}".format(validation["direct_ok_count"]))
    print("直接基线异常结果：{}".format(validation["direct_non_ok_count"]))
    print("知识库：{}".format(KB_PATH))
    print("配置文件：{}".format(config_path))
    print("\n下一步运行：python 02_build_rag_index.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("\n准备失败：{}".format(error), file=sys.stderr)
        sys.exit(1)
