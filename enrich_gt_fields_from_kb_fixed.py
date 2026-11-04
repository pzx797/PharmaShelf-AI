"""
根据知识库中的SKU信息，补全实例级真值表中的：
- gt_spec
- gt_manufacturer
- gt_barcode.

补全原则：
1.仅处理gt_sku_id非空的记录。
2.如果真值表存在label_status列，默认仅处理label_status=confirmed的记录。
3.默认只填空白单元格，不覆盖人工已经填写的内容。
4.知识库中同一SKU的某字段存在多个冲突值时，该字段不自动填充，并写入审计报告。
5.不覆盖原始真值表，输出新的Excel文件和审计工作表。

运行依赖：
    pip install pandas openpyxl

脚本名称：
    enrich_gt_fields_from_kb.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# =========================
# 1.路径配置
# =========================

KB_PATH = Path(r"E:\Desktop\Knowledge_Base\sku_kb_cleaned.xlsx")

GT_PATH = Path(
    r"E:\EdgeDownload\ultralytics-main\runs"
    r"\MSA_SKU_to_MRS_stage2_seed3_best_crops"
    r"\instances - gt.csv"
)

# 不覆盖原文件，统一输出为真正的.xlsx文件
OUTPUT_PATH = GT_PATH.with_name("instances_gt_v1_enriched.xlsx")


# =========================
# 2.处理策略
# =========================

# True：只有label_status=confirmed且gt_sku_id非空时才补全
# False：只要gt_sku_id非空就补全
REQUIRE_CONFIRMED = True

# False：保留真值表中已有的非空内容，只补空白
# True：知识库值覆盖真值表已有内容，不建议轻易开启
OVERWRITE_EXISTING = False

# 是否添加元数据来源列，便于论文中说明字段来自知识库映射
ADD_SOURCE_COLUMN = True
SOURCE_COLUMN = "gt_metadata_source"
SOURCE_TEXT = "knowledge_base_by_gt_sku_id"


# =========================
# 3.字段候选名称
# 已适配当前知识库表头：
# sku_id、barcode、generic_name、drug_name、raw_product_name、
# manufacturer、raw_specification、specification、category、
# raw_category、aliases、visual_keywords、shelf_positions、
# text_for_embedding
# =========================

COLUMN_CANDIDATES = {
    "sku_id": [
        "sku_id",
        "SKU_ID",
        "sku",
        "SKU",
        "商品编码",
        "药品编码",
        "商品ID",
        "SKU编码",
        "sku编码",
        "gt_sku_id",
    ],
    "spec": [
        "specification",  # 优先使用清洗后的规格字段
        "raw_specification",  # 清洗后字段不存在时作为备选
        "spec",
        "规格",
        "药品规格",
        "商品规格",
        "包装规格",
    ],
    "manufacturer": [
        "manufacturer",
        "生产厂家",
        "厂家",
        "生产企业",
        "制造商",
        "生产厂商",
        "企业名称",
    ],
    "barcode": [
        "barcode",
        "条码",
        "商品条码",
        "药品条码",
        "国际条码",
        "EAN",
        "ean",
    ],
}

TARGET_COLUMNS = {
    "spec": "gt_spec",
    "manufacturer": "gt_manufacturer",
    "barcode": "gt_barcode",
}


def clean_text(value: object) -> str:
    """将空值统一为空字符串，并清理首尾空格。."""
    if value is None:
        return ""

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""

    return text


def normalize_sku(value: object) -> str:
    """将SKU统一为字符串，避免Excel把整数SKU读取成2001057.0。 不对SKU补零，因为无法确定知识库是否使用前导零。.
    """
    text = clean_text(value)
    if not text:
        return ""

    # 仅移除典型的Excel整数小数尾缀
    text = re.sub(r"\.0+$", "", text)
    return text


def normalize_column_name(name: object) -> str:
    """用于宽松匹配表头。."""
    text = clean_text(name).lower()
    return re.sub(r"[\s_\-（）()\[\]【】/\\]+", "", text)


def find_column(
    columns: Iterable[object],
    candidates: Iterable[str],
    field_name: str,
) -> str:
    """从候选表头中自动查找实际列名。."""
    normalized_actual = {normalize_column_name(column): str(column) for column in columns}

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)
        if normalized_candidate in normalized_actual:
            return normalized_actual[normalized_candidate]

    available = "、".join(map(str, columns))
    raise KeyError(
        f"未在知识库中找到字段“{field_name}”。\n"
        f"当前知识库表头为：{available}\n"
        f"请在COLUMN_CANDIDATES中补充该字段的实际表头。"
    )


def detect_excel_file(path: Path) -> bool:
    """根据文件头判断是否为Excel工作簿。 兼容“扩展名是.csv，但内部实际是.xlsx”的情况。.
    """
    with path.open("rb") as file:
        signature = file.read(8)

    # xlsx是ZIP容器，通常以PK开头
    if signature.startswith(b"PK"):
        return True

    # 旧版xls复合文档文件头
    if signature.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        return True

    return path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}


def read_table(path: Path) -> pd.DataFrame:
    """自动读取Excel或真正的CSV，并尽量将字段保持为字符串。."""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    if detect_excel_file(path):
        return pd.read_excel(
            path,
            dtype=str,
            keep_default_na=False,
        )

    last_error: Exception | None = None

    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
                sep=None,
                engine="python",
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"无法读取CSV文件：{path}。最后一次错误：{last_error}")


def unique_nonempty_values(series: pd.Series) -> list[str]:
    """获取某列去空、去重后的值，并保持原始出现顺序。."""
    values: list[str] = []
    seen: set[str] = set()

    for raw_value in series.tolist():
        value = clean_text(raw_value)
        if value and value not in seen:
            seen.add(value)
            values.append(value)

    return values


def main() -> None:
    print("正在读取知识库：", KB_PATH)
    kb_df = read_table(KB_PATH)

    print("正在读取真值表：", GT_PATH)
    gt_df = read_table(GT_PATH)

    # 检查真值表关键字段
    if "gt_sku_id" not in gt_df.columns:
        raise KeyError("真值表中缺少gt_sku_id列，无法按SKU补全字段。")

    # 自动识别知识库字段
    kb_sku_col = find_column(
        kb_df.columns,
        COLUMN_CANDIDATES["sku_id"],
        "SKU编号",
    )
    kb_spec_col = find_column(
        kb_df.columns,
        COLUMN_CANDIDATES["spec"],
        "规格",
    )
    kb_manufacturer_col = find_column(
        kb_df.columns,
        COLUMN_CANDIDATES["manufacturer"],
        "生产厂家",
    )
    kb_barcode_col = find_column(
        kb_df.columns,
        COLUMN_CANDIDATES["barcode"],
        "条码",
    )

    print("\n知识库字段映射：")
    print(f"  SKU：{kb_sku_col}")
    print(f"  规格：{kb_spec_col}")
    print(f"  厂家：{kb_manufacturer_col}")
    print(f"  条码：{kb_barcode_col}")

    # 如果目标列不存在则创建
    for target_col in TARGET_COLUMNS.values():
        if target_col not in gt_df.columns:
            gt_df[target_col] = ""

    if ADD_SOURCE_COLUMN and SOURCE_COLUMN not in gt_df.columns:
        gt_df[SOURCE_COLUMN] = ""

    # SKU标准化
    kb_df["_normalized_sku"] = kb_df[kb_sku_col].map(normalize_sku)
    gt_df["_normalized_sku"] = gt_df["gt_sku_id"].map(normalize_sku)

    kb_df = kb_df[kb_df["_normalized_sku"] != ""].copy()

    # 构建知识库SKU映射，同时检测同一SKU的字段冲突
    kb_mapping: dict[str, dict[str, str]] = {}
    conflict_records: list[dict[str, str]] = []

    for sku, group in kb_df.groupby("_normalized_sku", sort=False):
        field_values = {
            "spec": unique_nonempty_values(group[kb_spec_col]),
            "manufacturer": unique_nonempty_values(group[kb_manufacturer_col]),
            "barcode": unique_nonempty_values(group[kb_barcode_col]),
        }

        kb_mapping[sku] = {}

        for field_name, values in field_values.items():
            if len(values) == 1:
                kb_mapping[sku][field_name] = values[0]
            elif len(values) == 0:
                kb_mapping[sku][field_name] = ""
            else:
                # 存在冲突时不自动选择，避免写入错误真值
                kb_mapping[sku][field_name] = ""
                conflict_records.append(
                    {
                        "sku_id": sku,
                        "field": field_name,
                        "conflicting_values": " | ".join(values),
                        "knowledge_base_rows": str(len(group)),
                    }
                )

    # 统计与审计记录
    filled_counts = {
        "gt_spec": 0,
        "gt_manufacturer": 0,
        "gt_barcode": 0,
    }

    eligible_rows = 0
    matched_rows = 0

    unmatched_records: list[dict[str, str]] = []
    mismatch_records: list[dict[str, str]] = []
    skipped_nonconfirmed_records: list[dict[str, str]] = []

    status_exists = "label_status" in gt_df.columns
    crop_id_exists = "crop_id" in gt_df.columns

    for index, row in gt_df.iterrows():
        sku = clean_text(row["_normalized_sku"])

        if not sku:
            continue

        crop_id = clean_text(row["crop_id"]) if crop_id_exists else str(index + 2)

        if REQUIRE_CONFIRMED and status_exists:
            label_status = clean_text(row["label_status"]).lower()

            if label_status != "confirmed":
                skipped_nonconfirmed_records.append(
                    {
                        "excel_row": str(index + 2),
                        "crop_id": crop_id,
                        "gt_sku_id": sku,
                        "label_status": clean_text(row["label_status"]),
                    }
                )
                continue

        eligible_rows += 1

        if sku not in kb_mapping:
            unmatched_records.append(
                {
                    "excel_row": str(index + 2),
                    "crop_id": crop_id,
                    "gt_sku_id": sku,
                    "reason": "知识库中未找到该SKU",
                }
            )
            continue

        matched_rows += 1
        row_filled = False

        for source_field, target_col in TARGET_COLUMNS.items():
            kb_value = clean_text(kb_mapping[sku].get(source_field, ""))
            current_value = clean_text(row[target_col])

            # 知识库字段为空或存在冲突时，不补全
            if not kb_value:
                continue

            # 当前真值字段为空时，直接补全
            if not current_value:
                gt_df.at[index, target_col] = kb_value
                filled_counts[target_col] += 1
                row_filled = True
                continue

            # 已有内容与知识库一致时，不处理
            if current_value == kb_value:
                continue

            # 真值表已有值与知识库不一致
            mismatch_records.append(
                {
                    "excel_row": str(index + 2),
                    "crop_id": crop_id,
                    "gt_sku_id": sku,
                    "field": target_col,
                    "current_gt_value": current_value,
                    "knowledge_base_value": kb_value,
                    "action": ("overwritten" if OVERWRITE_EXISTING else "kept_original"),
                }
            )

            if OVERWRITE_EXISTING:
                gt_df.at[index, target_col] = kb_value
                row_filled = True

        if ADD_SOURCE_COLUMN and row_filled:
            current_source = clean_text(gt_df.at[index, SOURCE_COLUMN])

            if not current_source:
                gt_df.at[index, SOURCE_COLUMN] = SOURCE_TEXT

    # 删除内部辅助列
    gt_df.drop(columns=["_normalized_sku"], inplace=True)

    # 汇总报告
    summary_df = pd.DataFrame(
        [
            ["真值表总实例数", len(gt_df)],
            ["gt_sku_id非空且满足处理条件的实例数", eligible_rows],
            ["在知识库中匹配到SKU的实例数", matched_rows],
            ["未在知识库中匹配到SKU的实例数", len(unmatched_records)],
            ["补全gt_spec的单元格数", filled_counts["gt_spec"]],
            [
                "补全gt_manufacturer的单元格数",
                filled_counts["gt_manufacturer"],
            ],
            ["补全gt_barcode的单元格数", filled_counts["gt_barcode"]],
            ["知识库SKU字段冲突数", len(conflict_records)],
            [
                "已有真值与知识库不一致的单元格数",
                len(mismatch_records),
            ],
            [
                "因label_status不是confirmed而跳过的实例数",
                len(skipped_nonconfirmed_records),
            ],
            ["是否覆盖已有非空值", OVERWRITE_EXISTING],
            ["是否要求label_status=confirmed", REQUIRE_CONFIRMED],
        ],
        columns=["item", "value"],
    )

    unmatched_df = pd.DataFrame(
        unmatched_records,
        columns=[
            "excel_row",
            "crop_id",
            "gt_sku_id",
            "reason",
        ],
    )

    conflicts_df = pd.DataFrame(
        conflict_records,
        columns=[
            "sku_id",
            "field",
            "conflicting_values",
            "knowledge_base_rows",
        ],
    )

    mismatches_df = pd.DataFrame(
        mismatch_records,
        columns=[
            "excel_row",
            "crop_id",
            "gt_sku_id",
            "field",
            "current_gt_value",
            "knowledge_base_value",
            "action",
        ],
    )

    skipped_df = pd.DataFrame(
        skipped_nonconfirmed_records,
        columns=[
            "excel_row",
            "crop_id",
            "gt_sku_id",
            "label_status",
        ],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 输出一个Excel工作簿，主表和审计记录分别放在不同工作表
    with pd.ExcelWriter(
        OUTPUT_PATH,
        engine="openpyxl",
    ) as writer:
        gt_df.to_excel(
            writer,
            sheet_name="instances_gt",
            index=False,
        )
        summary_df.to_excel(
            writer,
            sheet_name="enrichment_summary",
            index=False,
        )
        unmatched_df.to_excel(
            writer,
            sheet_name="unmatched_sku",
            index=False,
        )
        conflicts_df.to_excel(
            writer,
            sheet_name="kb_conflicts",
            index=False,
        )
        mismatches_df.to_excel(
            writer,
            sheet_name="value_mismatches",
            index=False,
        )
        skipped_df.to_excel(
            writer,
            sheet_name="skipped_nonconfirmed",
            index=False,
        )

    print("\n补全完成。")
    print(f"输出文件：{OUTPUT_PATH}")
    print(f"符合处理条件的实例数：{eligible_rows}")
    print(f"匹配到知识库SKU的实例数：{matched_rows}")
    print(f"补全gt_spec：{filled_counts['gt_spec']}个")
    print(f"补全gt_manufacturer：{filled_counts['gt_manufacturer']}个")
    print(f"补全gt_barcode：{filled_counts['gt_barcode']}个")
    print(f"未匹配SKU：{len(unmatched_records)}个")
    print(f"知识库字段冲突：{len(conflict_records)}个")
    print(f"已有值与知识库不一致：{len(mismatch_records)}个")

    if conflict_records:
        print("\n警告：知识库中存在同一SKU对应多个规格、厂家或条码的情况，冲突字段未自动补全。")

    if mismatch_records and not OVERWRITE_EXISTING:
        print("提示：已有真值与知识库不一致时，脚本保留了原值。请查看value_mismatches工作表。")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n运行失败：{exc}", file=sys.stderr)
        sys.exit(1)
