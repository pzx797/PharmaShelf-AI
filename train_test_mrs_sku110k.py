"""Train YOLO11n with SOSE + MAFF using a precision-first two-stage schedule once on SKU-110K, then evaluate on val and test."""

# 服务器训练   tmux new -s yolo_train
# 重新连接 tmux attach -t yolo_train
# 先按 Ctrl + b，松开，再按 d
# tmux ls

import re
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CFG = PROJECT_ROOT / "yolo11n-sose-maff.yaml"
DATA_CFG = Path(r"/hy-tmp/ultralytics-main/SKU110K_fixed.yaml")
PRETRAINED_WEIGHTS = Path(r"/hy-tmp/ultralytics-main/yolo11n.pt")

# 只训练和测试一次
SEED = 0

STAGE1_EPOCHS = 50
STAGE2_EPOCHS = 250

SAVE_ROOT = Path(r"/hy-tmp/ultralytics-main/runs/SKU_model_compare")


def transfer_yolo11n_to_sose_maff(model: YOLO, weights: Path) -> int:
    """Transfer YOLO11n weights while leaving SOSE and MAFF layers randomly initialized.

    SOSE is inserted at layer 11, moving original layers 11-22 to 12-23. MAFF is inserted before Detect at layers 24-26,
    moving original Detect layer 23 to 27.
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
    print("SOSE layer 11 and MAFF layers 24-26 remain new; original Neck/Detect shapes are preserved where possible.")
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
    "project": str(SAVE_ROOT),
}


def get_metrics(metrics):
    """Extract P, R, mAP50 and mAP50-95 from Ultralytics validation results."""
    result_dict = metrics.results_dict
    return {
        "map50": result_dict.get("metrics/mAP50(B)", 0.0),
        "map50_95": result_dict.get("metrics/mAP50-95(B)", 0.0),
        "precision": result_dict.get("metrics/precision(B)", 0.0),
        "recall": result_dict.get("metrics/recall(B)", 0.0),
    }


if __name__ == "__main__":
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("Training MRS (YOLO11n-SOSE-MAFF) on SKU-110K")
    print(f"Dataset: {DATA_CFG}")
    print(f"Save root: {SAVE_ROOT}")
    print(f"Seed: {SEED}")
    print(f"{'=' * 70}\n")

    # 1. 初始化模型并迁移 YOLO11n 预训练权重
    model = YOLO(MODEL_CFG)
    transfer_yolo11n_to_sose_maff(model, PRETRAINED_WEIGHTS)

    # 2. 第一阶段训练（冻结前11层）
    stage1_args = {
        **COMMON_TRAIN_ARGS,
        "epochs": STAGE1_EPOCHS,
        "freeze": 11,
    }
    model.train(
        **stage1_args,
        seed=SEED,
        name=f"yolo11n_SOSE_MAFF_precision_stage1_seed{SEED}",
    )

    stage1_best = Path(model.trainer.best)
    print(f"\nStage 1 best weights: {stage1_best}")

    model = YOLO(stage1_best)

    # 3. 第二阶段训练（全部可训练）
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
        name=f"yolo11n_SOSE_MAFF_precision_seed{SEED}",
    )

    stage2_best = Path(model.trainer.best)
    print(f"\nFinal best weights: {stage2_best}")

    final_model = YOLO(stage2_best)

    # 4. 在官方 val 集上评估
    print(f"\n{'=' * 70}")
    print("Evaluating final best.pt on SKU-110K VAL split")
    print(f"{'=' * 70}")

    val_metrics_obj = final_model.val(
        data=str(DATA_CFG),
        split="val",
        imgsz=960,
        batch=16,
        workers=0,
        device=0,
        project=str(SAVE_ROOT),
        name="MRS_SKU110K_val",
        plots=True,
    )
    val_result = get_metrics(val_metrics_obj)

    # 5. 在官方 test 集上测试
    print(f"\n{'=' * 70}")
    print("Testing final best.pt on SKU-110K TEST split")
    print(f"{'=' * 70}")

    test_metrics_obj = final_model.val(
        data=str(DATA_CFG),
        split="test",
        imgsz=960,
        batch=16,
        workers=0,
        device=0,
        project=str(SAVE_ROOT),
        name="MRS_SKU110K_test",
        plots=True,
    )
    test_result = get_metrics(test_metrics_obj)

    # 6. 控制台输出最终结果
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS")
    print(f"{'=' * 70}")

    print(
        f"VAL : "
        f"P={val_result['precision']:.4f}, "
        f"R={val_result['recall']:.4f}, "
        f"mAP50={val_result['map50']:.4f}, "
        f"mAP50-95={val_result['map50_95']:.4f}"
    )

    print(
        f"TEST: "
        f"P={test_result['precision']:.4f}, "
        f"R={test_result['recall']:.4f}, "
        f"mAP50={test_result['map50']:.4f}, "
        f"mAP50-95={test_result['map50_95']:.4f}"
    )

    print(f"Stage 1 best: {stage1_best}")
    print(f"Final best  : {stage2_best}")

    # 7. 保存训练/验证和测试结果
    save_path = SAVE_ROOT / "MRS_SKU110K_train_test_results.txt"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("MRS (YOLO11n-SOSE-MAFF) on SKU-110K\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: {DATA_CFG}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Stage1 epochs: {STAGE1_EPOCHS}\n")
        f.write(f"Stage2 epochs: {STAGE2_EPOCHS}\n")
        f.write(f"Total scheduled epochs: {STAGE1_EPOCHS + STAGE2_EPOCHS}\n")
        f.write(f"Stage1 best weights: {stage1_best}\n")
        f.write(f"Final best weights: {stage2_best}\n\n")

        f.write("Validation results (final best.pt on val split):\n")
        f.write(f"Precision: {val_result['precision']:.4f}\n")
        f.write(f"Recall: {val_result['recall']:.4f}\n")
        f.write(f"mAP50: {val_result['map50']:.4f}\n")
        f.write(f"mAP50-95: {val_result['map50_95']:.4f}\n\n")

        f.write("Test results (final best.pt on test split):\n")
        f.write(f"Precision: {test_result['precision']:.4f}\n")
        f.write(f"Recall: {test_result['recall']:.4f}\n")
        f.write(f"mAP50: {test_result['map50']:.4f}\n")
        f.write(f"mAP50-95: {test_result['map50_95']:.4f}\n")

    print(f"\nResults saved to: {save_path}")
