"""
Crop drug instances from final YOLO predict results and build an instance-level SKU labeling table.

Script name:
    crop_yolo_predict_to_sku_instances.py

Input:
    Original test images:
    /hy-tmp/ultralytics-main/datasets/MRSDataset/images/test

    YOLO predict result directory:
    /hy-tmp/ultralytics-main/runs/predict/MSA_SKU_to_MRS_stage2_seed3_best

Output:
    /hy-tmp/ultralytics-main/runs/MSA_SKU_to_MRS_stage2_seed3_best_crops/
        ├── crops/
        ├── instances.csv
        └── skipped_predictions.csv

Cropping rules:
    1. Use final YOLO predict results.
    2. Only crop boxes with confidence >= 0.25.
    3. Convert normalized YOLO bbox to valid pixel coordinates.
    4. Add 5% padding around each bbox.
    5. Use unique crop_id: image_name_box1, image_name_box2, ...
    6. Record all cropped instances in instances.csv.
    7. False positives, occlusion, partial boxes, and multi-object cases are not deleted;
       they are left as manual annotation fields in instances.csv.
"""

from pathlib import Path
import csv
import glob
import cv2


# =========================
# Basic configuration
# =========================

PREDICT_DIR = Path("/hy-tmp/ultralytics-main/runs/predict/MSA_SKU_to_MRS_stage2_seed3_best")
SAVE_ROOT = Path("/hy-tmp/ultralytics-main/runs")

CONF_THRES = 0.25
PADDING_RATIO = 0.05

# 原始 test 图片目录，必须是未画检测框的原图
ORIGINAL_IMAGES_DIR = Path("/hy-tmp/ultralytics-main/datasets/MRSDataset/images/test")

OUTPUT_DIR = SAVE_ROOT / "MSA_SKU_to_MRS_stage2_seed3_best_crops"
CROP_DIR = OUTPUT_DIR / "crops"

INSTANCES_CSV = OUTPUT_DIR / "instances.csv"
SKIPPED_CSV = OUTPUT_DIR / "skipped_predictions.csv"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_source_from_args_yaml(predict_dir: Path):
    """
    Try to read original source path from Ultralytics args.yaml.
    This is only a fallback. In this script, ORIGINAL_IMAGES_DIR is manually specified first.
    """
    args_yaml = predict_dir / "args.yaml"
    if not args_yaml.exists():
        return None

    try:
        import yaml
        with open(args_yaml, "r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        source = args.get("source", None)
        if source:
            return Path(str(source))
    except Exception:
        # Fallback: simple line parsing
        with open(args_yaml, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("source:"):
                    source = line.split("source:", 1)[1].strip().strip("'").strip('"')
                    if source:
                        return Path(source)

    return None


def collect_images_from_source(source):
    """
    Collect original images from:
    - a directory
    - a single image file
    - a txt file containing image paths
    - a glob pattern
    """
    image_paths = []

    if source is None:
        return image_paths

    source = Path(source)

    if source.is_dir():
        for p in source.rglob("*"):
            if p.suffix.lower() in IMAGE_SUFFIXES:
                image_paths.append(p)

    elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        image_paths.append(source)

    elif source.is_file() and source.suffix.lower() == ".txt":
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                p = Path(line.strip())
                if p.exists() and p.suffix.lower() in IMAGE_SUFFIXES:
                    image_paths.append(p)

    else:
        # Treat as glob pattern
        for p in glob.glob(str(source), recursive=True):
            p = Path(p)
            if p.exists() and p.suffix.lower() in IMAGE_SUFFIXES:
                image_paths.append(p)

    return sorted(image_paths)


def build_image_index(image_paths):
    """
    Build index by image stem.
    Example:
        IMG_001.jpg -> key: IMG_001
    """
    index = {}
    duplicate_stems = set()

    for p in image_paths:
        stem = p.stem
        if stem in index:
            duplicate_stems.add(stem)
        else:
            index[stem] = p

    if duplicate_stems:
        print(f"[Warning] Duplicate image stems found: {len(duplicate_stems)}")
        print("[Warning] The first matched image path will be used for duplicated stems.")

    return index


def get_original_image_index():
    """
    Priority:
    1. ORIGINAL_IMAGES_DIR manually configured
    2. source path from predict_dir/args.yaml
    3. fallback to images in PREDICT_DIR

    Note:
        fallback may use visualization images with boxes drawn on them.
        Since ORIGINAL_IMAGES_DIR is specified, normal execution will use original clean test images.
    """
    if ORIGINAL_IMAGES_DIR is not None:
        image_paths = collect_images_from_source(ORIGINAL_IMAGES_DIR)
        if image_paths:
            print(f"[Info] Using manually configured original images: {ORIGINAL_IMAGES_DIR}")
            return build_image_index(image_paths)

    source = read_source_from_args_yaml(PREDICT_DIR)
    if source is not None:
        image_paths = collect_images_from_source(source)
        if image_paths:
            print(f"[Info] Using original images from args.yaml source: {source}")
            return build_image_index(image_paths)

    # Fallback
    print("[Warning] Could not find original image source from ORIGINAL_IMAGES_DIR or args.yaml.")
    print("[Warning] Fallback: using images inside predict directory.")
    print("[Warning] These images may be visualization images with detection boxes drawn on them.")
    image_paths = collect_images_from_source(PREDICT_DIR)
    return build_image_index(image_paths)


def yolo_xywhn_to_xyxy_pixels(xc, yc, bw, bh, img_w, img_h):
    """
    Convert normalized YOLO xywh bbox to pixel xyxy bbox.
    """
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h
    return x1, y1, x2, y2


def add_padding_and_clip(x1, y1, x2, y2, img_w, img_h, padding_ratio=0.05):
    """
    Add padding and clip bbox to image boundary.
    """
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    x1p = max(0, int(round(x1 - pad_x)))
    y1p = max(0, int(round(y1 - pad_y)))
    x2p = min(img_w, int(round(x2 + pad_x)))
    y2p = min(img_h, int(round(y2 + pad_y)))

    return x1p, y1p, x2p, y2p


def parse_prediction_line(line):
    """
    Ultralytics save_txt + save_conf format:
        class x_center y_center width height confidence

    Returns:
        cls_id, xc, yc, bw, bh, conf
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


def main():
    labels_dir = PREDICT_DIR / "labels"

    if not PREDICT_DIR.exists():
        raise FileNotFoundError(f"Predict directory not found: {PREDICT_DIR}")

    if not labels_dir.exists():
        raise FileNotFoundError(
            f"Labels directory not found: {labels_dir}\n"
            "Please make sure YOLO predict was run with save_txt=True and save_conf=True."
        )

    if not ORIGINAL_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Original test image directory not found: {ORIGINAL_IMAGES_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    image_index = get_original_image_index()
    if not image_index:
        raise RuntimeError(
            "No source images found. Please check ORIGINAL_IMAGES_DIR."
        )

    label_files = sorted(labels_dir.glob("*.txt"))

    if not label_files:
        raise RuntimeError(f"No prediction txt files found in: {labels_dir}")

    instance_rows = []
    skipped_rows = []

    total_labels = 0
    total_crops = 0
    skipped_low_conf = 0
    skipped_invalid = 0
    skipped_missing_image = 0

    for label_file in label_files:
        image_stem = label_file.stem
        image_path = image_index.get(image_stem)

        if image_path is None:
            skipped_missing_image += 1
            skipped_rows.append({
                "source_image_stem": image_stem,
                "label_file": str(label_file),
                "reason": "missing_original_image",
                "raw_line": ""
            })
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            skipped_rows.append({
                "source_image_stem": image_stem,
                "label_file": str(label_file),
                "reason": "failed_to_read_image",
                "raw_line": ""
            })
            continue

        img_h, img_w = img.shape[:2]

        with open(label_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        box_order = 0

        for raw_line in lines:
            total_labels += 1

            parsed = parse_prediction_line(raw_line)
            if parsed is None:
                skipped_invalid += 1
                skipped_rows.append({
                    "source_image_stem": image_stem,
                    "label_file": str(label_file),
                    "reason": "invalid_label_format_or_missing_conf",
                    "raw_line": raw_line
                })
                continue

            cls_id, xc, yc, bw, bh, conf = parsed

            if conf < CONF_THRES:
                skipped_low_conf += 1
                skipped_rows.append({
                    "source_image_stem": image_stem,
                    "label_file": str(label_file),
                    "reason": f"low_conf_less_than_{CONF_THRES}",
                    "raw_line": raw_line
                })
                continue

            # Basic normalized bbox validation
            if bw <= 0 or bh <= 0:
                skipped_invalid += 1
                skipped_rows.append({
                    "source_image_stem": image_stem,
                    "label_file": str(label_file),
                    "reason": "non_positive_bbox_width_or_height",
                    "raw_line": raw_line
                })
                continue

            x1, y1, x2, y2 = yolo_xywhn_to_xyxy_pixels(xc, yc, bw, bh, img_w, img_h)

            # Clip original bbox first for validation
            x1_clip = max(0, min(img_w, int(round(x1))))
            y1_clip = max(0, min(img_h, int(round(y1))))
            x2_clip = max(0, min(img_w, int(round(x2))))
            y2_clip = max(0, min(img_h, int(round(y2))))

            if x1_clip >= x2_clip or y1_clip >= y2_clip:
                skipped_invalid += 1
                skipped_rows.append({
                    "source_image_stem": image_stem,
                    "label_file": str(label_file),
                    "reason": "invalid_pixel_bbox_after_clipping",
                    "raw_line": raw_line
                })
                continue

            x1p, y1p, x2p, y2p = add_padding_and_clip(
                x1_clip, y1_clip, x2_clip, y2_clip, img_w, img_h, PADDING_RATIO
            )

            if x1p >= x2p or y1p >= y2p:
                skipped_invalid += 1
                skipped_rows.append({
                    "source_image_stem": image_stem,
                    "label_file": str(label_file),
                    "reason": "invalid_padded_bbox",
                    "raw_line": raw_line
                })
                continue

            box_order += 1
            crop_id = f"{image_stem}_box{box_order}"
            crop_filename = f"{crop_id}.jpg"
            crop_path = CROP_DIR / crop_filename

            crop_img = img[y1p:y2p, x1p:x2p]

            if crop_img.size == 0:
                skipped_invalid += 1
                skipped_rows.append({
                    "source_image_stem": image_stem,
                    "label_file": str(label_file),
                    "reason": "empty_crop",
                    "raw_line": raw_line
                })
                continue

            ok = cv2.imwrite(str(crop_path), crop_img)
            if not ok:
                skipped_rows.append({
                    "source_image_stem": image_stem,
                    "label_file": str(label_file),
                    "reason": "failed_to_save_crop",
                    "raw_line": raw_line
                })
                continue

            total_crops += 1

            # Keep manual labeling fields blank for later annotation.
            instance_rows.append({
                "crop_id": crop_id,
                "box_order": box_order,
                "source_image": str(image_path),
                "source_image_name": image_path.name,
                "label_file": str(label_file),
                "crop_path": str(crop_path),
                "class_id": cls_id,
                "confidence": round(conf, 6),

                # YOLO normalized bbox
                "yolo_x_center": xc,
                "yolo_y_center": yc,
                "yolo_width": bw,
                "yolo_height": bh,

                # Original pixel bbox without padding
                "bbox_x1": x1_clip,
                "bbox_y1": y1_clip,
                "bbox_x2": x2_clip,
                "bbox_y2": y2_clip,

                # Padded pixel bbox used for crop
                "crop_x1": x1p,
                "crop_y1": y1p,
                "crop_x2": x2p,
                "crop_y2": y2p,
                "padding_ratio": PADDING_RATIO,

                # Manual SKU ground-truth annotation fields
                "gt_sku_id": "",
                "gt_drug_name": "",
                "gt_spec": "",
                "gt_manufacturer": "",
                "gt_barcode_or_approval_no": "",

                # Manual quality/status fields
                "label_status": "",      # confirmed / uncertain / false_positive / invalid
                "visibility": "",        # complete / partial / occluded / severe_occluded
                "readable": "",          # yes / no / partly
                "multi_object": "",      # yes / no
                "false_positive": "",    # yes / no
                "note": ""
            })

    # Write instances.csv
    if instance_rows:
        with open(INSTANCES_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(instance_rows[0].keys()))
            writer.writeheader()
            writer.writerows(instance_rows)
    else:
        # Create an empty file with basic header if no valid crops
        with open(INSTANCES_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "crop_id", "box_order", "source_image", "crop_path",
                "class_id", "confidence", "gt_sku_id", "gt_drug_name",
                "gt_spec", "gt_manufacturer", "label_status", "note"
            ])

    # Write skipped_predictions.csv
    if skipped_rows:
        with open(SKIPPED_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(skipped_rows[0].keys()))
            writer.writeheader()
            writer.writerows(skipped_rows)
    else:
        with open(SKIPPED_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["source_image_stem", "label_file", "reason", "raw_line"])

    print("=" * 80)
    print("Crop finished.")
    print(f"Original images dir: {ORIGINAL_IMAGES_DIR}")
    print(f"Predict dir:         {PREDICT_DIR}")
    print(f"Labels dir:          {labels_dir}")
    print(f"Output dir:          {OUTPUT_DIR}")
    print(f"Crop dir:            {CROP_DIR}")
    print(f"Instances CSV:       {INSTANCES_CSV}")
    print(f"Skipped CSV:         {SKIPPED_CSV}")
    print("-" * 80)
    print(f"Total prediction lines:      {total_labels}")
    print(f"Total cropped instances:     {total_crops}")
    print(f"Skipped low confidence:      {skipped_low_conf}")
    print(f"Skipped invalid bbox/format: {skipped_invalid}")
    print(f"Skipped missing image:       {skipped_missing_image}")
    print("=" * 80)


if __name__ == "__main__":
    main()