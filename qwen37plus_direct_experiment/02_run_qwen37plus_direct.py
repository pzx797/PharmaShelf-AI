"""Run Qwen3.7-Plus direct recognition on all prepared drug crop images.

Run order:
    1. python 01_prepare_direct_experiment.py
    2. python 02_run_qwen37plus_direct.py
    3. python 03_evaluate_direct_results.py

Features:
- Uses the OpenAI-compatible Alibaba Cloud Model Studio endpoint.
- Sends each local image as a Base64 data URL; no filename is sent to the model.
- Uses fixed prompts and fixed model parameters from config.yaml.
- Requests a JSON object and validates the returned schema.
- Supports safe breakpoint resume.
- Retries only API/system failures, never retries a poor model answer or format error.
- Does not save the model's hidden reasoning text.

Dependencies:
    pip install -U openai pandas pyyaml

Python compatibility: Python 3.8+
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    import yaml
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit(f"缺少依赖。请先执行：pip install -U openai pandas pyyaml\n原始错误：{exc}")


OUTPUT_ROOT = Path(r"E:\EdgeDownload\ultralytics-main\qwen37plus_direct_experiment")
RUN_NAME = "qwen37plus_direct_v1"
DEFAULT_CONFIG_PATH = OUTPUT_ROOT / "runs" / RUN_NAME / "config.yaml"

ALLOWED_DECISIONS = {"matched", "unknown", "multi_object", "non_drug"}
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
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件：{path}。请先运行01_prepare_direct_experiment.py")
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj)
    if not isinstance(data, dict):
        raise ValueError("config.yaml内容无效")
    return data


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"未找到提示词文件：{path}")
    return path.read_text(encoding="utf-8").strip()


def append_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(text.rstrip() + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    append_line(path, json.dumps(payload, ensure_ascii=False, default=str))


def safe_json_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        suffix = image_path.suffix.lower()
        fallback = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = fallback.get(suffix)
    if not mime_type:
        raise ValueError(f"不支持的图片格式：{image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def extract_first_json_object(text: str) -> str:
    """Extract the first balanced JSON object without altering its contents."""
    start = text.find("{")
    if start < 0:
        raise ValueError("回答中不存在JSON对象")

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


def parse_json_answer(content: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型回答为空")

    cleaned = strip_code_fence(content)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = json.loads(extract_first_json_object(cleaned))

    if not isinstance(parsed, dict):
        raise ValueError("模型输出不是JSON对象")
    return parsed


def nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"null", "none", "n/a", "未知", "无法确定"}:
        return None
    return text


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            text = nullable_text(item)
            if text is not None:
                result.append(text)
        return result
    text = nullable_text(value)
    return [] if text is None else [text]


def normalize_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence > 1.0 and confidence <= 100.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def validate_prediction(parsed: dict[str, Any], sample_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    returned_sample_id = nullable_text(parsed.get("sample_id"))
    result["sample_id"] = sample_id
    result["sample_id_returned"] = returned_sample_id

    decision = nullable_text(parsed.get("decision"))
    if decision:
        decision = decision.casefold()
    if decision not in ALLOWED_DECISIONS:
        raise ValueError("decision无效：{}".format(parsed.get("decision")))
    result["decision"] = decision

    for field in [
        "pred_drug_name",
        "pred_generic_name",
        "pred_specification",
        "pred_manufacturer",
        "pred_barcode_or_approval_no",
        "pred_sku_id",
        "reason",
    ]:
        result[field] = nullable_text(parsed.get(field))

    # Direct-VLM is not allowed to output a knowledge-base SKU.
    result["pred_sku_id"] = None
    result["confidence"] = normalize_confidence(parsed.get("confidence"))
    result["visible_text"] = normalize_string_list(parsed.get("visible_text"))
    result["package_features"] = normalize_string_list(parsed.get("package_features"))

    if decision != "matched":
        # Preserve partial visual fields, but identity fields must be null for a rejection.
        result["pred_drug_name"] = None
        result["pred_generic_name"] = None
        result["pred_specification"] = None
        result["pred_manufacturer"] = None
        result["pred_barcode_or_approval_no"] = None

    return result


def usage_values(completion: Any) -> tuple[int | None, int | None, int | None, int | None]:
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
    if reasoning_tokens is None and isinstance(usage, dict):
        details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = details.get("reasoning_tokens")

    return prompt_tokens, completion_tokens, total_tokens, reasoning_tokens


def read_existing_predictions(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as exc:
        raise RuntimeError(f"读取已有预测表失败：{exc}")
    for column in PREDICTION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[PREDICTION_COLUMNS]


def write_predictions_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(str(temp_path), str(path))


def completed_sample_ids(frame: pd.DataFrame) -> set:
    if frame.empty:
        return set()
    final_statuses = {"ok", "format_error", "input_error"}
    mask = frame["status"].isin(final_statuses)
    return set(frame.loc[mask, "sample_id"].astype(str))


def upsert_prediction(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> None:
    sample_id = str(new_row["sample_id"])
    for index, row in enumerate(rows):
        if str(row.get("sample_id", "")) == sample_id:
            rows[index] = new_row
            return
    rows.append(new_row)


def blank_prediction_row(run_id: str, sample_id: str, model: str) -> dict[str, Any]:
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
    # Exponential backoff with small jitter: 2, 4, 8... seconds.
    return min(30.0, (2.0**attempt) + random.uniform(0.0, 0.5))


def main() -> int:
    parser = argparse.ArgumentParser(description="运行Qwen3.7-Plus直接识别实验")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="config.yaml路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前N个待运行样本；0表示全部。可先用--limit 3试跑。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新运行已完成样本。正式实验中不要使用，除非整组结果作废并重新开始。",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_yaml(config_path)
    paths = config["paths"]
    api_cfg = config["api"]

    manifest_path = Path(paths["manifest_path"])
    predictions_path = Path(paths["predictions_path"])
    raw_responses_path = Path(paths["raw_responses_path"])
    api_errors_log = Path(paths["api_errors_log"])
    run_info_log = Path(paths["run_info_log"])
    system_prompt_path = Path(paths["system_prompt_path"])
    user_prompt_path = Path(paths["user_prompt_path"])

    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到输入清单：{manifest_path}")

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required_manifest_columns = {"sample_id", "crop_path", "include_in_inference"}
    missing_columns = required_manifest_columns - set(manifest.columns)
    if missing_columns:
        raise KeyError(f"input_manifest.csv缺少字段：{sorted(missing_columns)}")

    manifest = manifest[manifest["include_in_inference"].astype(str) != "0"].copy()
    manifest = manifest.sort_values("sample_id").reset_index(drop=True)

    system_prompt = read_text(system_prompt_path)
    user_prompt_template = read_text(user_prompt_path)

    api_key_env = str(api_cfg["api_key_env"])
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise OSError(f'未读取到环境变量{api_key_env}。PowerShell临时设置示例：\n$env:{api_key_env}="你的API-Key"')

    client = OpenAI(
        api_key=api_key,
        base_url=str(api_cfg["base_url"]),
        timeout=float(api_cfg.get("timeout_seconds", 180)),
        max_retries=0,  # Use the explicit experiment retry policy below.
    )

    model = str(api_cfg["model"])
    enable_thinking = bool(api_cfg.get("enable_thinking", True))
    temperature = float(api_cfg.get("temperature", 0.1))
    max_tokens = int(api_cfg.get("max_tokens", 1024))
    max_retries = int(api_cfg.get("max_retries", 3))
    vl_high_resolution_images = bool(api_cfg.get("vl_high_resolution_images", True))

    existing = read_existing_predictions(predictions_path)
    prediction_rows: list[dict[str, Any]] = existing.to_dict(orient="records")
    completed = set() if args.force else completed_sample_ids(existing)

    pending = manifest[~manifest["sample_id"].isin(completed)].copy()
    if args.limit and args.limit > 0:
        pending = pending.head(args.limit)

    run_id = "direct_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
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
        "manifest_total": len(manifest),
        "already_completed": len(completed),
        "pending_this_run": len(pending),
        "force": bool(args.force),
    }
    append_line(run_info_log, json.dumps(run_metadata, ensure_ascii=False))

    if pending.empty:
        print("没有待处理样本。现有预测结果已覆盖输入清单。")
        return 0

    print(f"本次运行ID：{run_id}")
    print(f"待处理样本：{len(pending)}")
    print(f"模型：{model}")
    print(f"思考模式：{enable_thinking}")
    print(f"高分辨率图像：{vl_high_resolution_images}")

    success_count = 0
    format_error_count = 0
    api_error_count = 0
    input_error_count = 0

    total_pending = len(pending)
    for position, record in enumerate(pending.to_dict(orient="records"), start=1):
        sample_id = str(record["sample_id"])
        image_path = Path(record["crop_path"])
        print(f"[{position}/{total_pending}] {sample_id}", end=" ", flush=True)

        if not image_path.exists() or not image_path.is_file():
            row = blank_prediction_row(run_id, sample_id, model)
            row.update(
                {
                    "status": "input_error",
                    "decision": "api_error",
                    "parse_error": f"图片不存在：{image_path}",
                }
            )
            upsert_prediction(prediction_rows, row)
            write_predictions_atomic(predictions_path, prediction_rows)
            append_line(
                api_errors_log,
                f"{now_iso()}\t{sample_id}\tinput_error\t{image_path}",
            )
            input_error_count += 1
            print("输入错误")
            continue

        try:
            data_url = image_to_data_url(image_path)
        except Exception as exc:
            row = blank_prediction_row(run_id, sample_id, model)
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
                f"{now_iso()}\t{sample_id}\tinput_error\t{exc}",
            )
            input_error_count += 1
            print("输入错误")
            continue

        user_prompt = user_prompt_template.replace("{sample_id}", sample_id)
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
                    f"{now_iso()}\t{sample_id}\tattempt={attempt}/{max_retries}\t{type(exc).__name__}: {exc}",
                )
                if attempt < max_retries:
                    time.sleep(get_retry_delay(attempt))

        if completion is None:
            row = blank_prediction_row(run_id, sample_id, model)
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
        prompt_tokens, completion_tokens, total_tokens, reasoning_tokens = usage_values(completion)

        # Deliberately omit reasoning_content from saved artifacts.
        raw_record = {
            "run_id": run_id,
            "sample_id": sample_id,
            "processed_at": now_iso(),
            "model_requested": model,
            "model_returned": response_model,
            "response_id": response_id,
            "finish_reason": finish_reason,
            "content": content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
            "latency_seconds": latency_seconds,
            "attempt_count": attempt_used,
            "reasoning_content_saved": False,
        }
        append_jsonl(raw_responses_path, raw_record)

        row = blank_prediction_row(run_id, sample_id, str(response_model))
        row.update(
            {
                "response_id": response_id or "",
                "finish_reason": finish_reason or "",
                "prompt_tokens": prompt_tokens if prompt_tokens is not None else "",
                "completion_tokens": completion_tokens if completion_tokens is not None else "",
                "total_tokens": total_tokens if total_tokens is not None else "",
                "reasoning_tokens": reasoning_tokens if reasoning_tokens is not None else "",
                "latency_seconds": latency_seconds,
                "attempt_count": attempt_used,
            }
        )

        try:
            parsed = parse_json_answer(content)
            validated = validate_prediction(parsed, sample_id)
            row.update(
                {
                    "status": "ok",
                    "decision": validated["decision"],
                    "pred_drug_name": validated["pred_drug_name"] or "",
                    "pred_generic_name": validated["pred_generic_name"] or "",
                    "pred_specification": validated["pred_specification"] or "",
                    "pred_manufacturer": validated["pred_manufacturer"] or "",
                    "pred_barcode_or_approval_no": validated["pred_barcode_or_approval_no"] or "",
                    "pred_sku_id": "",
                    "confidence": "" if validated["confidence"] is None else validated["confidence"],
                    "visible_text_json": json.dumps(validated["visible_text"], ensure_ascii=False),
                    "package_features_json": json.dumps(validated["package_features"], ensure_ascii=False),
                    "reason": validated["reason"] or "",
                    "parse_error": "",
                }
            )
            success_count += 1
            print("完成：{}".format(validated["decision"]))
        except Exception as exc:
            row.update(
                {
                    "status": "format_error",
                    "decision": "format_error",
                    "parse_error": f"{type(exc).__name__}: {exc}",
                }
            )
            format_error_count += 1
            print("格式错误")

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
        "raw_responses_path": str(raw_responses_path),
    }
    append_line(run_info_log, json.dumps(finish_metadata, ensure_ascii=False))

    print("\n本次运行结束。")
    print(f"成功：{success_count}")
    print(f"格式错误：{format_error_count}")
    print(f"API失败：{api_error_count}")
    print(f"输入错误：{input_error_count}")
    print(f"预测表：{predictions_path}")
    print("\n确认没有api_error后，运行：python 03_evaluate_direct_results.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断。已完成结果已保存，可直接重新运行以断点续跑。")
        sys.exit(130)
    except Exception as error:
        print(f"\n运行失败：{error}", file=sys.stderr)
        sys.exit(1)
