"""
Draw box index labels on YOLO prediction visualization images.

Script name:
    draw_box_index_on_predict_images.py

Purpose:
    According to the order of YOLO prediction labels and the crop order,
    draw box1, box2, box3, ... on the predicted visualization images.

Input:
    YOLO predict result directory:
    /hy-tmp/ultralytics-main/runs/predict/MSA_SKU_to_MRS_stage2_seed3_best

    Cropped instance directory:
    /hy-tmp/ultralytics-main/runs/MSA_SKU_to_MRS_stage2_seed3_best_crops/crops

Output:
    /hy-tmp/ultralytics-main/runs/MSA_SKU_to_MRS_stage2_seed3_best_box_indexed/
        ├── images/
        └── box_index.csv

Important:
    The numbering rule is consistent with the crop script:
        - read prediction txt line by line
        - keep boxes with confidence >= 0.25
        - skip invalid boxes
        - box_order starts from 1 for each image
        - crop_id = image_name_box1, image_name_box2, ...
"""

import csv
from pathlib import Path

import cv2

# =========================
# Basic configuration
# =========================

PREDICT_DIR = Path("/hy-tmp/ultralytics-main/runs/predict/MSA_SKU_to_MRS_stage2_seed3_best")
SAVE_ROOT = Path("/hy-tmp/ultralytics-main/runs")

# 裁剪好的小图目录，用于检查 crop_id 是否存在
CROP_DIR = Path("/hy-tmp/ultralytics-main/runs/MSA_SKU_to_MRS_stage2_seed3_best_crops/crops")

OUTPUT_DIR = SAVE_ROOT / "MSA_SKU_to_MRS_stage2_seed3_best_box_indexed"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"
BOX_INDEX_CSV = OUTPUT_DIR / "box_index.csv"

CONF_THRES = 0.25

IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def find_image_by_stem(image_dir: Path, stem: str):
    """Find image file by stem in PREDICT_DIR.

    Examples:
        KZJ_0090_20250215_100123.txt
        -> KZJ_0090_20250215_100123.jpg.
    """
    for suffix in IMAGE_SUFFIXES:
        p = image_dir / f"{stem}{suffix}"
        if p.exists():
            return p

    # fallback: recursive search
    for suffix in IMAGE_SUFFIXES:
        matches = list(image_dir.rglob(f"{stem}{suffix}"))
        if matches:
            return matches[0]

    return None


def parse_prediction_line(line):
    """Ultralytics save_txt + save_conf format: class x_center y_center width height confidence.
    """
    parts = line.strip().split()
    if len(parts) < 6:
        return None

    cls_id = int(float(parts[0]))
    xc = float(parts[1])
    yc = float(parts[2])
    bw = float(parts[3])
    bh = float(parts[4])
    conf = float(parts[5])

    return cls_id, xc, yc, bw, bh, conf


def yolo_xywhn_to_xyxy_pixels(xc, yc, bw, bh, img_w, img_h):
    """Convert normalized YOLO bbox to pixel bbox."""
    x1 = round((xc - bw / 2.0) * img_w)
    y1 = round((yc - bh / 2.0) * img_h)
    x2 = round((xc + bw / 2.0) * img_w)
    y2 = round((yc + bh / 2.0) * img_h)

    x1 = max(0, min(img_w, x1))
    y1 = max(0, min(img_h, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))

    return x1, y1, x2, y2


def draw_label(img, text, x1, y1):
    """Draw a readable box index label."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    # Put label near top-left of bbox.
    label_x = x1
    label_y = y1 - 8

    # If the label is outside the top boundary, put it inside the bbox.
    if label_y - text_h - baseline < 0:
        label_y = y1 + text_h + baseline + 8

    # Background rectangle coordinates
    bg_x1 = label_x
    bg_y1 = label_y - text_h - baseline
    bg_x2 = label_x + text_w + 8
    bg_y2 = label_y + baseline

    bg_x1 = max(0, bg_x1)
    bg_y1 = max(0, bg_y1)
    bg_x2 = min(img.shape[1], bg_x2)
    bg_y2 = min(img.shape[0], bg_y2)

    # Red background + white text
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), -1)
    cv2.putText(img, text, (label_x + 4, label_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    labels_dir = PREDICT_DIR / "labels"

    if not PREDICT_DIR.exists():
        raise FileNotFoundError(f"Predict directory not found: {PREDICT_DIR}")

    if not labels_dir.exists():
        raise FileNotFoundError(
            f"Labels directory not found: {labels_dir}\n"
            "Please make sure YOLO predict was run with save_txt=True and save_conf=True."
        )

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))

    if not label_files:
        raise RuntimeError(f"No prediction txt files found in: {labels_dir}")

    index_rows = []

    total_images = 0
    total_boxes = 0
    skipped_low_conf = 0
    skipped_invalid = 0
    skipped_missing_image = 0
    missing_crop_count = 0

    for label_file in label_files:
        image_stem = label_file.stem
        predict_image_path = find_image_by_stem(PREDICT_DIR, image_stem)

        if predict_image_path is None:
            skipped_missing_image += 1
            print(f"[Warning] Missing predicted image for label: {label_file}")
            continue

        img = cv2.imread(str(predict_image_path))
        if img is None:
            print(f"[Warning] Failed to read image: {predict_image_path}")
            continue

        img_h, img_w = img.shape[:2]

        with open(label_file, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        box_order = 0

        for raw_line in lines:
            parsed = parse_prediction_line(raw_line)
            if parsed is None:
                skipped_invalid += 1
                continue

            cls_id, xc, yc, bw, bh, conf = parsed

            if conf < CONF_THRES:
                skipped_low_conf += 1
                continue

            if bw <= 0 or bh <= 0:
                skipped_invalid += 1
                continue

            x1, y1, x2, y2 = yolo_xywhn_to_xyxy_pixels(xc, yc, bw, bh, img_w, img_h)

            if x1 >= x2 or y1 >= y2:
                skipped_invalid += 1
                continue

            box_order += 1
            total_boxes += 1

            box_name = f"box{box_order}"
            crop_id = f"{image_stem}_{box_name}"
            crop_path = CROP_DIR / f"{crop_id}.jpg"

            crop_exists = crop_path.exists()
            if not crop_exists:
                missing_crop_count += 1

            draw_label(img, box_name, x1, y1)

            index_rows.append(
                {
                    "source_image_stem": image_stem,
                    "predict_image": str(predict_image_path),
                    "label_file": str(label_file),
                    "box_order": box_order,
                    "box_name": box_name,
                    "crop_id": crop_id,
                    "crop_path": str(crop_path),
                    "crop_exists": "yes" if crop_exists else "no",
                    "class_id": cls_id,
                    "confidence": round(conf, 6),
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "yolo_x_center": xc,
                    "yolo_y_center": yc,
                    "yolo_width": bw,
                    "yolo_height": bh,
                }
            )

        output_image_path = OUTPUT_IMAGE_DIR / predict_image_path.name
        ok = cv2.imwrite(str(output_image_path), img)
        if not ok:
            print(f"[Warning] Failed to save indexed image: {output_image_path}")
        else:
            total_images += 1

    # Save box_index.csv
    if index_rows:
        with open(BOX_INDEX_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)
    else:
        with open(BOX_INDEX_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "source_image_stem",
                    "predict_image",
                    "label_file",
                    "box_order",
                    "box_name",
                    "crop_id",
                    "crop_path",
                    "crop_exists",
                    "class_id",
                    "confidence",
                    "bbox_x1",
                    "bbox_y1",
                    "bbox_x2",
                    "bbox_y2",
                ]
            )

    print("=" * 80)
    print("Box index drawing finished.")
    print(f"Predict dir:          {PREDICT_DIR}")
    print(f"Labels dir:           {labels_dir}")
    print(f"Crop dir:             {CROP_DIR}")
    print(f"Output image dir:     {OUTPUT_IMAGE_DIR}")
    print(f"Box index CSV:        {BOX_INDEX_CSV}")
    print("-" * 80)
    print(f"Processed images:     {total_images}")
    print(f"Indexed boxes:        {total_boxes}")
    print(f"Skipped low conf:     {skipped_low_conf}")
    print(f"Skipped invalid:      {skipped_invalid}")
    print(f"Missing pred image:   {skipped_missing_image}")
    print(f"Missing crop files:   {missing_crop_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
