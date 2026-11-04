"""Train YOLO11n with the SOSE small-object shallow enhancement module."""

from pathlib import Path
import re

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CFG = PROJECT_ROOT / "yolo11n-sose.yaml"
DATA_CFG = PROJECT_ROOT / "MRSDataset.yaml"  # Update this YAML if the dataset location differs.
PRETRAINED_WEIGHTS = Path(r"E:\CODE\ultralytics-main\yolo11n.pt")  # COCO-pretrained YOLO11n checkpoint.
SEED = 0  # Set to 1, 2, 3, or 4 for subsequent random-seed experiments.


def transfer_yolo11n_to_sose(model: YOLO, weights: Path) -> int:
    """Transfer YOLO11n COCO weights while leaving the newly added SOSE layer randomly initialized.

    SOSE is inserted at layer 11, so every original Neck/Detect layer from 11 onward moves forward by one index.
    This explicit remapping preserves pretrained Backbone, Neck, and Detect features before dataset-specific training.
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
        target_index = source_index if source_index <= 10 else source_index + 1
        target_key = f"model.{target_index}.{suffix}"
        if target_key in target_state and target_state[target_key].shape == value.shape:
            transferred[target_key] = value

    model.model.load_state_dict(transferred, strict=False)
    # Model.train() rebuilds a one-class model from model.yaml. A nonempty ckpt makes it use this transferred
    # model as the source, retaining all compatible features and naturally reinitializing the one-class Detect head.
    model.ckpt = {"model": model.model}
    print(f"Transferred {len(transferred)}/{len(target_state)} YOLO11n COCO state items; SOSE layer 11 remains new.")
    return len(transferred)


if __name__ == "__main__":
    model = YOLO(MODEL_CFG)
    transfer_yolo11n_to_sose(model, PRETRAINED_WEIGHTS)
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
        patience=100,
        pretrained=False,  # We use the explicit layer-aware transfer above instead of Ultralytics' default loader.
        seed=SEED,
        name=f"yolo11n_SOSE_seed{SEED}",
    )
