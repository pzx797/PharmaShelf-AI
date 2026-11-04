"""
Fine-tune MSA-YOLO11 with SKU-pretrained weights on MRSDataset.
Run 5 seeds, evaluate each seed on test split, and calculate mean ± std.

实验流程：
1. 加载MSA-YOLO11在SKU-110K上预训练得到的best.pt；
2. 在健之佳MRSDataset上进行两阶段微调；
3. 每个seed训练完成后，在test集上测试一次；
4. 统计5次test结果的平均值和标准差；
5. 所有结果保存到：
   /hy-tmp/ultralytics-main/runs/Fine-tune_MSA-YOLO_sku_to_mrs
"""

# 服务器训练   tmux new -s yolo_train
# 重新连接     tmux attach -t yolo_train
# 暂时退出     先按Ctrl+b，松开，再按d
# 查看会话     tmux ls

import csv
from pathlib import Path
from statistics import mean, stdev

from ultralytics import YOLO

# =========================
# 路径配置
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_CFG = PROJECT_ROOT / "MRSDataset.yaml"

# MSA-YOLO11在SKU-110K上预训练得到的权重
PRETRAINED_WEIGHTS = Path("/hy-tmp/ultralytics-main/runs/MSA-YOLO_mrs_pretrained/yolo11n_MSA_SKU_final/weights/best.pt")

# 本次“SKU预训练权重→健之佳微调→test五种子评估”的结果保存目录
SAVE_ROOT = Path("/hy-tmp/ultralytics-main/runs/Fine-tune_MSA-YOLO_sku_to_mrs")

SEEDS = [0, 1, 2, 3, 4]

# 两阶段微调轮数
STAGE1_EPOCHS = 50
STAGE2_EPOCHS = 250


# =========================
# 工具函数
# =========================
def check_paths():
    """检查必要文件是否存在。."""
    if not DATA_CFG.is_file():
        raise FileNotFoundError(f"MRSDataset.yaml was not found: {DATA_CFG}")

    if not PRETRAINED_WEIGHTS.is_file():
        raise FileNotFoundError(f"SKU pretrained MSA weight was not found: {PRETRAINED_WEIGHTS}")

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)


def get_best_weight(model: YOLO) -> Path:
    """获取当前训练阶段的best.pt；如果不存在，则回退到last.pt。."""
    best_path = Path(model.trainer.best)

    if best_path.is_file():
        return best_path

    last_path = Path(model.trainer.save_dir) / "weights" / "last.pt"
    if last_path.is_file():
        print(f"Warning: best.pt not found, using last.pt instead: {last_path}")
        return last_path

    raise FileNotFoundError(f"Neither best.pt nor last.pt was found in {Path(model.trainer.save_dir) / 'weights'}")


def extract_metrics(metrics):
    """从Ultralytics验证结果中提取P、R、mAP50、mAP50-95。."""
    result_dict = getattr(metrics, "results_dict", {}) or {}

    precision = result_dict.get("metrics/precision(B)", None)
    recall = result_dict.get("metrics/recall(B)", None)
    map50 = result_dict.get("metrics/mAP50(B)", None)
    map50_95 = result_dict.get("metrics/mAP50-95(B)", None)

    # 兼容不同Ultralytics版本
    if precision is None:
        precision = float(metrics.box.mp)
    if recall is None:
        recall = float(metrics.box.mr)
    if map50 is None:
        map50 = float(metrics.box.map50)
    if map50_95 is None:
        map50_95 = float(metrics.box.map)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "map50": float(map50),
        "map50_95": float(map50_95),
    }


def metric_mean_std(all_results, key):
    """计算均值和标准差。."""
    values = [r[key] for r in all_results]
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def save_results(all_results):
    """保存每个seed的test结果，以及5次实验的均值和标准差。."""
    csv_path = SAVE_ROOT / "MSA_SKU_to_MRS_5seeds_test_results.csv"
    txt_path = SAVE_ROOT / "MSA_SKU_to_MRS_5seeds_summary.txt"

    p_mean, p_std = metric_mean_std(all_results, "precision")
    r_mean, r_std = metric_mean_std(all_results, "recall")
    map50_mean, map50_std = metric_mean_std(all_results, "map50")
    map50_95_mean, map50_95_std = metric_mean_std(all_results, "map50_95")

    # 保存CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "stage1_best",
                "stage2_best",
                "test_save_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    # 保存TXT汇总
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MSA-YOLO11: SKU pretrained weights -> MRSDataset fine-tuning -> test evaluation\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"Pretrained weights:\n{PRETRAINED_WEIGHTS}\n\n")
        f.write(f"Dataset config:\n{DATA_CFG}\n\n")
        f.write(f"Save root:\n{SAVE_ROOT}\n\n")

        f.write("Per-seed test results:\n")
        f.writelines(
            f"seed={r['seed']}, "
            f"P={r['precision']:.6f}, "
            f"R={r['recall']:.6f}, "
            f"mAP50={r['map50']:.6f}, "
            f"mAP50-95={r['map50_95']:.6f}\n"
            for r in all_results
        )

        f.write("\nMean ± Std over 5 seeds:\n")
        f.write(f"Precision = {p_mean:.6f} ± {p_std:.6f}\n")
        f.write(f"Recall = {r_mean:.6f} ± {r_std:.6f}\n")
        f.write(f"mAP50 = {map50_mean:.6f} ± {map50_std:.6f}\n")
        f.write(f"mAP50-95 = {map50_95_mean:.6f} ± {map50_95_std:.6f}\n")

    print("\n" + "=" * 90)
    print("Final 5-seed test summary")
    print("=" * 90)
    print(f"Precision = {p_mean:.6f} ± {p_std:.6f}")
    print(f"Recall = {r_mean:.6f} ± {r_std:.6f}")
    print(f"mAP50 = {map50_mean:.6f} ± {map50_std:.6f}")
    print(f"mAP50-95 = {map50_95_mean:.6f} ± {map50_95_std:.6f}")
    print(f"\nCSV saved to: {csv_path}")
    print(f"Summary saved to: {txt_path}")


# =========================
# 基础训练参数
# =========================
COMMON_TRAIN_ARGS = {
    "data": str(DATA_CFG),
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
    # 基础数据增强，保持和前面实验一致
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
    # 不早停，保证每个seed训练轮数一致
    "patience": 0,
    # 当前模型已经从SKU预训练权重加载，不再额外加载官方预训练权重
    "pretrained": False,
    # 训练过程中只用val集选best.pt，test集只在最后单独评估
    "val": True,
    "plots": True,
    "save": True,
    "project": str(SAVE_ROOT),
}


COMMON_TEST_ARGS = {
    "data": str(DATA_CFG),
    "split": "test",
    "workers": 0,
    "imgsz": 960,
    "batch": 16,
    "device": 0,
    "plots": True,
    "project": str(SAVE_ROOT),
}


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    check_paths()

    all_results = []

    for seed in SEEDS:
        print("\n" + "=" * 90)
        print(f"Fine-tuning MSA-YOLO11 with seed={seed}")
        print("=" * 90)

        # -------------------------------------------------
        # 1. 加载SKU预训练后的MSA-YOLO11权重
        # -------------------------------------------------
        model = YOLO(str(PRETRAINED_WEIGHTS))

        # -------------------------------------------------
        # 2. 第一阶段微调：冻结前11层
        # -------------------------------------------------
        stage1_name = f"MSA_SKU_to_MRS_stage1_seed{seed}"

        stage1_args = {
            **COMMON_TRAIN_ARGS,
            "epochs": STAGE1_EPOCHS,
            "freeze": 11,
            "close_mosaic": 50,
        }

        print(f"\nStage 1 training: seed={seed}, freeze=11, epochs={STAGE1_EPOCHS}")
        model.train(
            **stage1_args,
            seed=seed,
            name=stage1_name,
            exist_ok=True,
        )

        stage1_best = get_best_weight(model)
        print(f"Stage 1 best weight: {stage1_best}")

        # -------------------------------------------------
        # 3. 第二阶段微调：全部解冻训练
        # -------------------------------------------------
        model = YOLO(str(stage1_best))

        stage2_name = f"MSA_SKU_to_MRS_stage2_seed{seed}"

        stage2_args = {
            **COMMON_TRAIN_ARGS,
            "epochs": STAGE2_EPOCHS,
            "lr0": 0.006,
            "close_mosaic": 80,
        }

        print(f"\nStage 2 training: seed={seed}, freeze=None, epochs={STAGE2_EPOCHS}")
        model.train(
            **stage2_args,
            seed=seed,
            name=stage2_name,
            exist_ok=True,
        )

        stage2_best = get_best_weight(model)
        print(f"Stage 2 best weight: {stage2_best}")

        # -------------------------------------------------
        # 4. 使用test集进行最终测试
        # -------------------------------------------------
        print(f"\nTesting on MRSDataset test split: seed={seed}")

        final_model = YOLO(str(stage2_best))
        test_name = f"MSA_SKU_to_MRS_test_seed{seed}"

        metrics = final_model.val(
            **COMMON_TEST_ARGS,
            name=test_name,
            exist_ok=True,
        )

        metric_values = extract_metrics(metrics)

        test_save_dir = SAVE_ROOT / test_name

        one_result = {
            "seed": seed,
            "precision": metric_values["precision"],
            "recall": metric_values["recall"],
            "map50": metric_values["map50"],
            "map50_95": metric_values["map50_95"],
            "stage1_best": str(stage1_best),
            "stage2_best": str(stage2_best),
            "test_save_dir": str(test_save_dir),
        }

        all_results.append(one_result)

        print(
            f"\nSeed {seed} test results: "
            f"P={one_result['precision']:.6f}, "
            f"R={one_result['recall']:.6f}, "
            f"mAP50={one_result['map50']:.6f}, "
            f"mAP50-95={one_result['map50_95']:.6f}"
        )

    # -------------------------------------------------
    # 5. 统计5次test结果并保存
    # -------------------------------------------------
    save_results(all_results)
