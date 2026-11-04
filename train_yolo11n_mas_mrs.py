"""Train YOLO11n with SOSE + MAFF using a precision-first two-stage schedule, run 5 seeds, evaluate and average results."""

# 服务器训练   tmux new -s yolo_train
# 重新连接 tmux attach -t yolo_train
# 先按 Ctrl + b，松开，再按 d
# tmux ls

from pathlib import Path
import re

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CFG = PROJECT_ROOT / "yolo11n-sose-maff.yaml"
DATA_CFG = PROJECT_ROOT / "MRSDataset.yaml"
PRETRAINED_WEIGHTS = Path(r"/hy-tmp/ultralytics-main/yolo11n.pt")
SEEDS = [0, 1, 2, 3, 4]
STAGE1_EPOCHS = 50
STAGE2_EPOCHS = 250


def transfer_yolo11n_to_sose_maff(model: YOLO, weights: Path) -> int:
    """Transfer YOLO11n weights while leaving SOSE and MAFF layers randomly initialized.

    SOSE is inserted at layer 11, moving original layers 11-22 to 12-23.
    MAFF is inserted before Detect at layers 24-26, moving original Detect layer 23 to 27.
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


# 基础训练参数（不含种子）
COMMON_TRAIN_ARGS = dict(
    data=DATA_CFG,
    workers=0,
    imgsz=960,
    batch=16,
    device=0,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_bias_lr=0.01,
    cos_lr=False,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,
    degrees=5.0,
    translate=0.05,
    scale=0.15,
    shear=0.0,
    perspective=0.0,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.3,
    close_mosaic=50,
    patience=0,
    pretrained=False,
    project="/hy-tmp/ultralytics-main/runs",  # 所有训练结果保存在此目录
)


if __name__ == "__main__":
    all_results = []  # 存储每个种子的指标

    for seed in SEEDS:
        print(f"\n{'='*60}\nTraining with seed = {seed}\n{'='*60}")

        # 1. 初始化模型并迁移权重
        model = YOLO(MODEL_CFG)
        transfer_yolo11n_to_sose_maff(model, PRETRAINED_WEIGHTS)

        # 2. 第一阶段训练（冻结前11层）
        stage1_args = {**COMMON_TRAIN_ARGS, "epochs": STAGE1_EPOCHS, "freeze": 11}
        model.train(
            **stage1_args,
            seed=seed,
            name=f"yolo11n_SOSE_MAFF_precision_stage1_seed{seed}",
        )
        stage1_best = Path(model.trainer.best)
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
            seed=seed,
            name=f"yolo11n_SOSE_MAFF_precision_seed{seed}",
        )
        stage2_best = Path(model.trainer.best)
        final_model = YOLO(stage2_best)

        # 4. 在验证集上评估
        metrics = final_model.val()
        result_dict = metrics.results_dict  # 包含所有指标
        map50 = result_dict.get("metrics/mAP50(B)", 0.0)
        map50_95 = result_dict.get("metrics/mAP50-95(B)", 0.0)
        precision = result_dict.get("metrics/precision(B)", 0.0)
        recall = result_dict.get("metrics/recall(B)", 0.0)

        all_results.append({
            "seed": seed,
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
        })

        print(f"Seed {seed} results: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}, "
              f"P={precision:.4f}, R={recall:.4f}")

    # 5. 计算平均值
    avg_map50 = sum(r["map50"] for r in all_results) / len(all_results)
    avg_map50_95 = sum(r["map50_95"] for r in all_results) / len(all_results)
    avg_precision = sum(r["precision"] for r in all_results) / len(all_results)
    avg_recall = sum(r["recall"] for r in all_results) / len(all_results)

    # 6. 保存结果
    save_path = Path("/hy-tmp/ultralytics-main/runs/avg_results.txt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("Average results over 5 seeds (0-4):\n")
        f.write(f"mAP50: {avg_map50:.4f}\n")
        f.write(f"mAP50-95: {avg_map50_95:.4f}\n")
        f.write(f"Precision: {avg_precision:.4f}\n")
        f.write(f"Recall: {avg_recall:.4f}\n\n")
        f.write("Individual results:\n")
        for r in all_results:
            f.write(f"Seed {r['seed']}: mAP50={r['map50']:.4f}, "
                    f"mAP50-95={r['map50_95']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}\n")

    print(f"\nAverage results saved to {save_path}")