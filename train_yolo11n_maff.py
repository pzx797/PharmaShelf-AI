"""Train YOLO11n with the MAFF multi-scale attention fusion module."""

from pathlib import Path
import re

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CFG = PROJECT_ROOT / "yolo11n-maff.yaml"
DATA_CFG = PROJECT_ROOT / "MRSDataset.yaml"
PRETRAINED_WEIGHTS = Path(r"E:\CODE\ultralytics-main\yolo11n.pt")
SEED = 0


def transfer_yolo11n_to_maff(model: YOLO, weights: Path) -> int:
    """Transfer YOLO11n weights while leaving MAFF layers randomly initialized.

    MAFF is inserted before Detect, so original layers 0-22 keep their indices and original Detect layer 23 moves to
    target layer 26. This preserves the pretrained Detect head for a fairer MAFF-only ablation.
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
        if source_index <= 22:
            target_index = source_index
        elif source_index == 23:
            target_index = 26
        else:
            continue
        target_key = f"model.{target_index}.{suffix}"
        if target_key in target_state and target_state[target_key].shape == value.shape:
            transferred[target_key] = value

    model.model.load_state_dict(transferred, strict=False)
    model.ckpt = {"model": model.model}
    print(f"Transferred {len(transferred)}/{len(target_state)} YOLO11n state items; MAFF layers 23-25 remain new.")
    return len(transferred)


if __name__ == "__main__":
    model = YOLO(MODEL_CFG)
    transfer_yolo11n_to_maff(model, PRETRAINED_WEIGHTS)
    model.train(
        data=DATA_CFG,
        workers=0,
        epochs=300,
        imgsz=960,
        batch=4,
        device=0,
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
        patience=0,
        pretrained=False,
        seed=SEED,
        name=f"yolo11n_MAFF_seed{SEED}",
    )
