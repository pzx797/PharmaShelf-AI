 # -*- coding: utf-8 -*-
"""构建药品知识库的本地RAG检索索引。

检索方法：
- 使用字符级TF-IDF表示中文药品名称、规格、厂家、别名和条码等字段；
- 保存知识库记录、向量器和稀疏矩阵；
- 推理阶段再叠加名称、规格、厂家和条码的精确匹配加权。

该脚本只读取知识库，不读取人工真值表。

运行：
    python 02_build_rag_index.py

依赖：
    pip install pandas openpyxl pyyaml scikit-learn joblib

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
from typing import Any, Dict, List

try:
    import joblib
    import numpy as np
    import pandas as pd
    import yaml
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as exc:
    raise SystemExit(
        "缺少依赖。请先执行："
        "pip install pandas openpyxl pyyaml scikit-learn joblib\n"
        "原始错误：{}".format(exc)
    )


OUTPUT_ROOT = Path(r"E:\EdgeDownload\ultralytics-main\qwen37plus_rag_experiment")
RUN_NAME = "qwen37plus_rag_v1"
DEFAULT_CONFIG_PATH = OUTPUT_ROOT / "runs" / RUN_NAME / "config.yaml"

KB_REQUIRED_COLUMNS = {
    "sku_id",
    "drug_name",
    "generic_name",
    "raw_product_name",
    "manufacturer",
    "raw_specification",
    "specification",
    "category",
    "aliases",
    "visual_keywords",
    "text_for_embedding",
}


# -----------------------------------------------------------------------------
# 通用辅助函数
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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """以UTF-8写入格式化JSON文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def clean_cell(value: Any) -> str:
    """将Excel单元格转换为稳定文本，避免条码和SKU出现科学计数法。"""
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
    if re.fullmatch(r"\d+(?:\.\d+)?[eE][+-]?\d+", text):
        try:
            number = float(text)
            if number.is_integer():
                return "{:.0f}".format(number)
        except ValueError:
            pass
    return text


def normalize_text(value: Any) -> str:
    """执行字符标准化，用于精确匹配和包含匹配。"""
    text = clean_cell(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("×", "x").replace("✕", "x").replace("＊", "x")
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r"[，,。.;；:：'\"“”‘’/\\_\-·•()（）\[\]【】{}]+", "", text)
    return text


def combine_weighted_fields(row: pd.Series) -> str:
    """按字段重要性构造知识库检索文本。"""
    drug_name = clean_cell(row.get("drug_name"))
    generic_name = clean_cell(row.get("generic_name"))
    raw_product_name = clean_cell(row.get("raw_product_name"))
    specification = clean_cell(row.get("specification"))
    raw_specification = clean_cell(row.get("raw_specification"))
    manufacturer = clean_cell(row.get("manufacturer"))
    barcode = clean_cell(row.get("barcode"))
    sku_id = clean_cell(row.get("sku_id"))
    category = clean_cell(row.get("category"))
    aliases = clean_cell(row.get("aliases"))
    visual_keywords = clean_cell(row.get("visual_keywords"))
    text_for_embedding = clean_cell(row.get("text_for_embedding"))

    # 名称和条码重复写入，使字符级TF-IDF更重视身份字段。
    parts = [
        drug_name,
        drug_name,
        drug_name,
        generic_name,
        generic_name,
        raw_product_name,
        raw_product_name,
        specification,
        specification,
        raw_specification,
        manufacturer,
        manufacturer,
        barcode,
        barcode,
        sku_id,
        aliases,
        aliases,
        visual_keywords,
        category,
        text_for_embedding,
    ]
    return " ".join(part for part in parts if part)


def build_record(row: pd.Series, kb_row_index: int) -> Dict[str, str]:
    """将知识库行转换为推理阶段可用的标准记录。"""
    fields = [
        "sku_id",
        "barcode",
        "generic_name",
        "drug_name",
        "raw_product_name",
        "manufacturer",
        "raw_specification",
        "specification",
        "category",
        "raw_category",
        "aliases",
        "visual_keywords",
        "shelf_positions",
        "text_for_embedding",
    ]
    record = {field: clean_cell(row.get(field)) for field in fields}
    record["kb_row_index"] = str(kb_row_index)
    record["retrieval_text"] = combine_weighted_fields(row)
    record["norm_drug_name"] = normalize_text(record["drug_name"])
    record["norm_generic_name"] = normalize_text(record["generic_name"])
    record["norm_specification"] = normalize_text(record["specification"])
    record["norm_manufacturer"] = normalize_text(record["manufacturer"])
    record["norm_barcode"] = normalize_text(record["barcode"])
    record["norm_aliases"] = normalize_text(record["aliases"])
    return record


def main() -> int:
    """读取知识库、训练TF-IDF向量器并保存索引。"""
    parser = argparse.ArgumentParser(description="构建药品知识库RAG检索索引")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="RAG实验config.yaml路径",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_yaml(config_path)
    paths = config["paths"]
    retrieval_cfg = config["retrieval"]
    tfidf_cfg = retrieval_cfg["tfidf"]

    kb_path = Path(paths["knowledge_base_path"])
    index_path = Path(paths["retrieval_index_path"])
    records_path = Path(paths["retrieval_records_path"])
    metadata_path = Path(paths["retrieval_metadata_path"])
    sheet_name = config.get("data", {}).get("knowledge_base_sheet", 0)

    if not kb_path.exists():
        raise FileNotFoundError("未找到知识库：{}".format(kb_path))

    print("正在读取知识库：{}".format(kb_path))
    kb_df = pd.read_excel(
        kb_path,
        sheet_name=sheet_name,
        engine="openpyxl",
        dtype=object,
    )

    missing_columns = KB_REQUIRED_COLUMNS - set(kb_df.columns)
    if missing_columns:
        raise KeyError(
            "知识库缺少字段：{}\n当前表头：{}".format(
                sorted(missing_columns), "、".join(map(str, kb_df.columns.tolist()))
            )
        )

    records: List[Dict[str, str]] = []
    skipped_missing_sku = 0
    for dataframe_index, row in kb_df.iterrows():
        record = build_record(row, int(dataframe_index))
        if not record["sku_id"]:
            skipped_missing_sku += 1
            continue
        if not record["retrieval_text"].strip():
            continue
        records.append(record)

    if not records:
        raise RuntimeError("知识库没有可用于构建索引的有效记录")

    sku_values = [record["sku_id"] for record in records]
    duplicate_skus = pd.Series(sku_values)[pd.Series(sku_values).duplicated()].tolist()
    if duplicate_skus:
        raise ValueError(
            "知识库sku_id存在重复，例如：{}".format(duplicate_skus[:10])
        )

    texts = [record["retrieval_text"] for record in records]
    ngram_range = tuple(int(value) for value in tfidf_cfg["ngram_range"])

    print("正在构建字符级TF-IDF索引，记录数：{}".format(len(records)))
    vectorizer = TfidfVectorizer(
        analyzer=str(tfidf_cfg.get("analyzer", "char")),
        ngram_range=ngram_range,
        min_df=1,
        max_features=int(tfidf_cfg.get("max_features", 120000)),
        sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", True)),
        norm=str(tfidf_cfg.get("norm", "l2")),
        lowercase=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "index_version": "1.0",
        "created_at": now_iso(),
        "method": retrieval_cfg.get("method", "char_tfidf_field_boost_v1"),
        "vectorizer": vectorizer,
        "matrix": matrix,
        "records": records,
        "config_path": str(config_path),
        "knowledge_base_path": str(kb_path),
    }
    joblib.dump(bundle, index_path, compress=3)

    # 同时导出可检查的知识库标准记录，不依赖joblib即可人工审阅。
    export_df = pd.DataFrame(records)
    export_df.to_csv(records_path, index=False, encoding="utf-8-sig")

    metadata = {
        "created_at": now_iso(),
        "method": retrieval_cfg.get("method", "char_tfidf_field_boost_v1"),
        "knowledge_base_path": str(kb_path),
        "knowledge_base_rows": int(len(kb_df)),
        "indexed_rows": int(len(records)),
        "skipped_missing_sku": int(skipped_missing_sku),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "ngram_range": list(ngram_range),
        "top_k": int(retrieval_cfg.get("top_k", 5)),
        "ground_truth_used": False,
        "index_path": str(index_path),
        "records_path": str(records_path),
    }
    write_json(metadata_path, metadata)

    print("\n索引构建完成。")
    print("有效知识库记录：{}".format(len(records)))
    print("跳过缺失SKU记录：{}".format(skipped_missing_sku))
    print("向量矩阵形状：{}".format(matrix.shape))
    print("索引文件：{}".format(index_path))
    print("记录文件：{}".format(records_path))
    print("\n下一步运行：python 03_run_qwen37plus_rag.py --limit 3")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("\n索引构建失败：{}".format(error), file=sys.stderr)
        sys.exit(1)
