"""Train YOLO11n with MSA using a precision-first two-stage schedule, run once, evaluate results."""

# 服务器训练   tmux new -s yolo_train
# 重新连接 tmux attach -t yolo_train
# 先按 Ctrl + b，松开，再按 d
# tmux ls

import re
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CFG = PROJECT_ROOT / "yolo11n-msa.yaml"
DATA_CFG = PROJECT_ROOT / "SKU110K_fixed.yaml"
PRETRAINED_WEIGHTS = Path(r"/hy-tmp/ultralytics-main/yolo11n.pt")

SEED = 0
STAGE1_EPOCHS = 50
STAGE2_EPOCHS = 250

SAVE_PROJECT = "/hy-tmp/ultralytics-main/runs/MSA-YOLO_mrs_pretrained"


def transfer_yolo11n_to_msa(model: YOLO, weights: Path) -> int:
    """Transfer YOLO11n weights while leaving MSA layers randomly initialized.

    MSA is inserted at layer 11, moving original layers 11-22 to 12-23. MSA fusion layers are inserted before Detect at
    layers 24-26, moving original Detect layer 23 to 27.
    """
    if not weights.is_file():
        raise FileNotFoundError(f"COCO pretrained weight file was not found: {weights}")

    source_state = YOLO(str(weights)).model.state_dict()
    target_state = model.model.state_dict()
    transferred = {}
    pattern = re.compile(r"^model\.(\d+)\.(.+)$")

    for source_key, value in source_state.items():
        match = pattern.match(source_key)
        if not match:
            continue

        source_index, suffix = int(match.group(1)), match.group(2)

        if source_index <= 10:
            target_index = source_index
        elif source_index <= 22:
            target_index = source_index + 1
        elif source_index == 23:
            target_index = 27
        else:
            continue

        target_key = f"model.{target_index}.{suffix}"

        if target_key in target_state and target_state[target_key].shape == value.shape:
            transferred[target_key] = value

    model.model.load_state_dict(transferred, strict=False)
    model.ckpt = {"model": model.model}

    skipped = len(target_state) - len(transferred)
    print(f"Transferred {len(transferred)}/{len(target_state)} YOLO11n state items; {skipped} new or reshaped items.")
    print(
        "MSA layer 11 and MSA fusion layers 24-26 remain new; original Neck/Detect shapes are preserved where possible."
    )

    return len(transferred)


# 基础训练参数
COMMON_TRAIN_ARGS = {
    "data": DATA_CFG,
    "workers": 0,
    "imgsz": 960,
    "batch": 16,
    "device": 0,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_bias_lr": 0.01,
    "cos_lr": False,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "degrees": 5.0,
    "translate": 0.05,
    "scale": 0.15,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.3,
    "close_mosaic": 50,
    "patience": 0,
    "pretrained": False,
    "project": SAVE_PROJECT,
}


if __name__ == "__main__":
    print(f"\n{'=' * 60}\nTraining MSA on SKU110K with seed = {SEED}\n{'=' * 60}")

    # 1. 初始化模型并迁移YOLO11n权重
    model = YOLO(MODEL_CFG)
    transfer_yolo11n_to_msa(model, PRETRAINED_WEIGHTS)

    # 2. 第一阶段训练：冻结前11层
    stage1_args = {
        **COMMON_TRAIN_ARGS,
        "epochs": STAGE1_EPOCHS,
        "freeze": 11,
    }

    model.train(
        **stage1_args,
        seed=SEED,
        name="yolo11n_MSA_SKU_stage1",
    )

    stage1_best = Path(model.trainer.best)
    model = YOLO(stage1_best)

    # 3. 第二阶段训练：全部可训练
    stage2_args = {
        **COMMON_TRAIN_ARGS,
        "epochs": STAGE2_EPOCHS,
        "freeze": None,
        "lr0": 0.006,
        "close_mosaic": 80,
    }

    model.train(
        **stage2_args,
        seed=SEED,
        name="yolo11n_MSA_SKU_final",
    )

    stage2_best = Path(model.trainer.best)
    final_model = YOLO(stage2_best)

    # 4. 在验证集上评估
    metrics = final_model.val(data=DATA_CFG)
    result_dict = metrics.results_dict

    map50 = result_dict.get("metrics/mAP50(B)", 0.0)
    map50_95 = result_dict.get("metrics/mAP50-95(B)", 0.0)
    precision = result_dict.get("metrics/precision(B)", 0.0)
    recall = result_dict.get("metrics/recall(B)", 0.0)

    print(f"\nFinal results: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}, P={precision:.4f}, R={recall:.4f}")

    # 5. 保存结果
    save_path = Path(SAVE_PROJECT) / "MSA_SKU_results.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("MSA-YOLO11 training results on SKU110K_fixed.yaml\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Stage1 epochs: {STAGE1_EPOCHS}\n")
        f.write(f"Stage2 epochs: {STAGE2_EPOCHS}\n")
        f.write(f"Stage1 best: {stage1_best}\n")
        f.write(f"Stage2 best: {stage2_best}\n\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"mAP50: {map50:.4f}\n")
        f.write(f"mAP50-95: {map50_95:.4f}\n")

    print(f"\nResults saved to {save_path}")
    print(f"Best model saved at {stage2_best}")
