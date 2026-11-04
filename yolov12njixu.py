from ultralytics import YOLO

LAST_PT = (
    "/hy-tmp/ultralytics-main/runs/SKU_model_compare/"
    "SKU110K_YOLOv12n_300e/weights/last.pt"
)

model = YOLO(LAST_PT)

model.train(
    resume=True,
    epochs=300,
    patience=0,
    device=0,
)