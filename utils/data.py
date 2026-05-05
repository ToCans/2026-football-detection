# Imports
import os
import cv2
import json
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"),
    label_extension: str = ".txt",
    seed: int = 42,
    copy: bool = True,
) -> dict:
    """
    Splits images and their corresponding labels from a source directory into
    train, valid, and test subsets.
 
    Expected source structure:
        source_dir/
            images/   (or any flat folder of image files)
            labels/   (label files with the same stem as the images)
 
    Output structure:
        output_dir/
            train/images/  train/labels/
            valid/images/  valid/labels/
            test/images/   test/labels/
 
    Args:
        source_dir:        Path to the directory containing 'images/' and 'labels/' subdirs.
        output_dir:        Path where the split dataset will be written.
        train_ratio:       Fraction of data used for training (default 0.7).
        val_ratio:         Fraction of data used for validation (default 0.2).
        test_ratio:        Fraction of data used for testing (default 0.1).
        image_extensions:  Tuple of valid image file extensions (case-insensitive).
        label_extension:   Extension of label files (default '.txt' for YOLO format).
        seed:              Random seed for reproducibility.
        copy:              If True, copy files; if False, move them.
 
    Returns:
        A dict with split counts: {"train": n, "valid": n, "test": n, "skipped": n}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "train_ratio + val_ratio + test_ratio must equal 1.0"
    )
 
    source = Path(source_dir)
    images_dir = source / "images"
    labels_dir = source / "labels"
 
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
 
    # Collect all image files
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    )
 
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
 
    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(image_files)
 
    # Compute split indices
    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
 
    splits = {
        "train": image_files[:n_train],
        "valid": image_files[n_train:n_train + n_val],
        "test":  image_files[n_train + n_val:],
    }
 
    # Create output subdirectories
    output = Path(output_dir)
    for split in splits:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)
 
    transfer = shutil.copy2 if copy else shutil.move
    counts = {"train": 0, "valid": 0, "test": 0, "skipped": 0}
 
    for split_name, files in splits.items():
        for img_path in files:
            label_path = labels_dir / (img_path.stem + label_extension)
 
            if not label_path.exists():
                print(f"[WARNING] No label found for {img_path.name} — skipping.")
                counts["skipped"] += 1
                continue
 
            dest_img   = output / split_name / "images" / img_path.name
            dest_label = output / split_name / "labels" / label_path.name
 
            transfer(img_path, dest_img)
            transfer(label_path, dest_label)
            counts[split_name] += 1
 
    total = counts["train"] + counts["valid"] + counts["test"]
    print(f"\nDataset split complete ({total} pairs):")
    print(f"  train : {counts['train']}")
    print(f"  valid : {counts['valid']}")
    print(f"  test  : {counts['test']}")
    if counts["skipped"]:
        print(f"  skipped (no label): {counts['skipped']}")
 
    return counts

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def yolo_to_pixel(box, img_w, img_h):
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def pixel_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def read_labels(label_path):
    labels = []
    if not Path(label_path).exists():
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                box = tuple(float(v) for v in parts[1:])
                labels.append((cls, *box))
    return labels


def write_labels(label_path, labels):
    with open(label_path, "w") as f:
        for cls, cx, cy, bw, bh in labels:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def random_scale(crop, scale_range):
    s = random.uniform(*scale_range)
    h, w = crop.shape[:2]
    new_w = max(4, int(w * s))
    new_h = max(4, int(h * s))
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def random_brightness(crop, delta):
    d = random.randint(-delta, delta)
    return np.clip(crop.astype(np.int16) + d, 0, 255).astype(np.uint8)


def paste_crop(image, crop, x1, y1):
    img_h, img_w = image.shape[:2]
    ch, cw = crop.shape[:2]
    x2 = min(x1 + cw, img_w)
    y2 = min(y1 + ch, img_h)
    if x2 <= x1 or y2 <= y1:
        return None
    image[y1:y2, x1:x2] = crop[: y2 - y1, : x2 - x1]
    return x1, y1, x2, y2


def boxes_overlap(box_new, existing_boxes, img_w, img_h, iou_thresh=0.05):
    nx1, ny1, nx2, ny2 = yolo_to_pixel(box_new[1:], img_w, img_h)
    for _, cx, cy, bw, bh in existing_boxes:
        ex1, ey1, ex2, ey2 = yolo_to_pixel((cx, cy, bw, bh), img_w, img_h)
        ix1, iy1 = max(nx1, ex1), max(ny1, ey1)
        ix2, iy2 = min(nx2, ex2), min(ny2, ey2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        new_area = (nx2 - nx1) * (ny2 - ny1)
        if inter > 0 and new_area > 0 and inter / new_area > iou_thresh:
            return True
    return False

def compute_placement_bounds(labels_dir, ball_class):
    """
    Scan every label file in a partition and collect the normalised centre
    coordinates (cx, cy) of every ball annotation.  Returns the observed
    min/max as a dict so placement of augmented balls stays inside the same
    spatial region as real ones.
 
    Falls back to [0, 1] x [0, 1] (full image) if no balls are found.
    """
    cx_vals, cy_vals = [], []
 
    label_files = sorted(Path(labels_dir).glob("*.txt"))
    for lf in label_files:
        for cls, cx, cy, bw, bh in read_labels(lf):
            if cls == ball_class:
                cx_vals.append(cx)
                cy_vals.append(cy)
 
    if not cx_vals:
        print("  [WARN] No ball annotations found — using full-image placement range.")
        return {"cx_min": 0.0, "cx_max": 1.0, "cy_min": 0.0, "cy_max": 1.0}
 
    bounds = {
        "cx_min": min(cx_vals),
        "cx_max": max(cx_vals),
        "cy_min": min(cy_vals),
        "cy_max": max(cy_vals),
    }
    print(
        f"  [BOUNDS] cx=[{bounds['cx_min']:.3f}, {bounds['cx_max']:.3f}]  "
        f"cy=[{bounds['cy_min']:.3f}, {bounds['cy_max']:.3f}]  "
        f"(from {len(cx_vals)} ball annotations)"
    )
    return bounds

# ──────────────────────────────────────────────
# Per-image augmentation
# ──────────────────────────────────────────────
def augment_image(image_path, label_path, out_image_path, out_label_path,
                  ball_class, num_copies, placement_bounds, brighntess, scale, max_attempts):
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, "unreadable"
 
    img_h, img_w = image.shape[:2]
    labels = read_labels(label_path)
 
    # Crop all ball instances from this image
    ball_crops = []
    for cls, cx, cy, bw, bh in labels:
        if cls != ball_class:
            continue
        x1, y1, x2, y2 = yolo_to_pixel((cx, cy, bw, bh), img_w, img_h)
        if x2 > x1 and y2 > y1:
            ball_crops.append(image[y1:y2, x1:x2].copy())
 
    if not ball_crops:
        # No ball in this image — copy unchanged
        shutil.copy(image_path, out_image_path)
        if Path(label_path).exists():
            shutil.copy(label_path, out_label_path)
        else:
            write_labels(out_label_path, [])
        return 0, "no_ball"
 
    new_labels = list(labels)
    aug_image  = image.copy()
    added      = 0
 
    for _ in range(num_copies):
        # Pick a random ball crop and apply augmentations
        crop = random.choice(ball_crops).copy()
        crop = random_brightness(crop, brighntess)
        crop = random_scale(crop, scale)
        ch, cw = crop.shape[:2]
 
        # Derive pixel placement window from normalised bounds.
        # The bounds describe where ball *centres* appear; we offset by
        # half the crop size so the centre lands inside the valid region.
        px_min = max(0,         int(placement_bounds["cx_min"] * img_w) - cw // 2)
        px_max = min(img_w - cw, int(placement_bounds["cx_max"] * img_w) - cw // 2)
        py_min = max(0,         int(placement_bounds["cy_min"] * img_h) - ch // 2)
        py_max = min(img_h - ch, int(placement_bounds["cy_max"] * img_h) - ch // 2)
 
        # Graceful fallback: if the crop is larger than the allowed window,
        # open the range back up to avoid an invalid randint call.
        if px_max < px_min:
            px_min, px_max = 0, max(0, img_w - cw)
        if py_max < py_min:
            py_min, py_max = 0, max(0, img_h - ch)
 
        # Try to place it without overlapping existing boxes
        for _ in range(max_attempts):
            px = random.randint(px_min, px_max)
            py = random.randint(py_min, py_max)
            candidate = (ball_class, *pixel_to_yolo(px, py, px + cw, py + ch, img_w, img_h))
            if not boxes_overlap(candidate, new_labels, img_w, img_h):
                result = paste_crop(aug_image, crop, px, py)
                if result:
                    new_labels.append((ball_class, *pixel_to_yolo(*result, img_w, img_h)))
                    added += 1
                break
 
    cv2.imwrite(str(out_image_path), aug_image)
    write_labels(out_label_path, new_labels)
    return added, "ok"

# ──────────────────────────────────────────────
# Partition processing
# ──────────────────────────────────────────────

def process_partition(partition, src_root, dst_root, ball_class, num_copies, brighntess, scale, max_attempts):
    src_images = Path(src_root) / partition / "images"
    src_labels = Path(src_root) / partition / "labels"

    if not src_images.exists():
        print(f"  [SKIP] '{partition}' partition not found — skipping.")
        return

    dst_images = Path(dst_root) / partition / "images"
    dst_labels = Path(dst_root) / partition / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in src_images.iterdir() if p.suffix.lower() in exts])

    # Scan all label files once to learn where balls actually appear
    placement_bounds = compute_placement_bounds(src_labels, ball_class)


    total_added   = 0
    no_ball_count = 0

    with tqdm(image_paths, desc=f"  {partition:>5}", unit="img", colour="green") as pbar:
        for img_path in pbar:
            lbl_path     = src_labels / (img_path.stem + ".txt")
            out_img_path = dst_images / img_path.name
            out_lbl_path = dst_labels / (img_path.stem + ".txt")

            added, status = augment_image(
                img_path, lbl_path,
                out_img_path, out_lbl_path,
                ball_class, num_copies, placement_bounds, brighntess, scale, max_attempts
            )

            total_added   += added
            no_ball_count += int(status == "no_ball")
            pbar.set_postfix(added=total_added, no_ball=no_ball_count)

    print(f"         -> {len(image_paths)} images | +{total_added} ball annotations | {no_ball_count} images without ball")

def compute_transform(orig_w, orig_h, target_w, target_h, mode):
    """Returns (new_size, scale_x, scale_y, pad_left, pad_top)."""
    if mode == "stretch":
        return (target_w, target_h), target_w / orig_w, target_h / orig_h, 0, 0
    elif mode == "fit":
        scale  = min(target_w / orig_w, target_h / orig_h)
        sw, sh = int(round(orig_w * scale)), int(round(orig_h * scale))
        pl, pt = (target_w - sw) // 2, (target_h - sh) // 2
        return (target_w, target_h), scale, scale, pl, pt
    elif mode == "crop":
        scale  = max(target_w / orig_w, target_h / orig_h)
        sw, sh = int(round(orig_w * scale)), int(round(orig_h * scale))
        pl, pt = (target_w - sw) // 2, (target_h - sh) // 2
        return (target_w, target_h), scale, scale, pl, pt
    else:
        raise ValueError(f"Unknown RESIZE_MODE: {mode}")


def resize_image(img, target_w, target_h, mode, pad_color, resample):
    orig_w, orig_h = img.size
    _, sx, sy, pl, pt = compute_transform(orig_w, orig_h, target_w, target_h, mode)
    if mode == "stretch":
        return img.resize((target_w, target_h), resample)
    elif mode == "fit":
        sw, sh = int(round(orig_w * sx)), int(round(orig_h * sy))
        canvas = Image.new(img.mode, (target_w, target_h), pad_color)
        canvas.paste(img.resize((sw, sh), resample), (pl, pt))
        return canvas
    elif mode == "crop":
        sw, sh = int(round(orig_w * sx)), int(round(orig_h * sy))
        scaled = img.resize((sw, sh), resample)
        return scaled.crop((-pl, -pt, -pl + target_w, -pt + target_h))


def rescale_bbox(bbox, sx, sy, pl, pt, iw, ih):
    x, y, w, h = bbox
    x1, y1 = x * sx + pl,       y * sy + pt
    x2, y2 = (x + w) * sx + pl, (y + h) * sy + pt
    x1, y1 = max(0, min(x1, iw)), max(0, min(y1, ih))
    x2, y2 = max(0, min(x2, iw)), max(0, min(y2, ih))
    return [x1, y1, x2 - x1, y2 - y1]


def rescale_segmentation(segmentation, sx, sy, pl, pt, iw, ih):
    new_seg = []
    for polygon in segmentation:
        poly = []
        for i, v in enumerate(polygon):
            poly.append(max(0, min(v * sx + pl if i % 2 == 0 else v * sy + pt,
                                   iw if i % 2 == 0 else ih)))
        new_seg.append(poly)
    return new_seg


def rescale_keypoints(keypoints, sx, sy, pl, pt, iw, ih):
    new_kps = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
        if v == 0:
            new_kps.extend([0, 0, 0])
        else:
            new_kps.extend([max(0, min(x * sx + pl, iw)),
                            max(0, min(y * sy + pt, ih)), v])
    return new_kps


def process_split(split, input_root, output_root,
                  target_w, target_h, mode, pad_color, resample, ann_filename):
    in_img_dir   = Path(input_root)  / split
    in_ann_path  = Path(input_root)  / split / ann_filename
    out_img_dir  = Path(output_root) / split
    out_ann_path = Path(output_root) / split / ann_filename

    if not in_ann_path.exists():
        print(f"  [{split}] Annotation file not found, skipping: {in_ann_path}")
        return None

    out_img_dir.mkdir(parents=True, exist_ok=True)

    with open(in_ann_path) as f:
        coco = json.load(f)

    transforms = {}
    new_images  = []
    missing     = 0

    for img_rec in tqdm(coco["images"], desc=f"  [{split}] images", leave=False):
        src = in_img_dir  / img_rec["file_name"]
        dst = out_img_dir / img_rec["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            missing += 1
            continue

        with Image.open(src) as img:
            orig_w, orig_h = img.size
            result = resize_image(img, target_w, target_h, mode, pad_color, resample)
            if result is not None:
                result.save(dst)

        _, sx, sy, pl, pt = compute_transform(orig_w, orig_h, target_w, target_h, mode)
        transforms[img_rec["id"]] = (sx, sy, pl, pt)

        new_rec = dict(img_rec)
        new_rec["width"]  = target_w
        new_rec["height"] = target_h
        new_images.append(new_rec)

    new_annotations = []
    for ann in tqdm(coco["annotations"], desc=f"  [{split}] anns ", leave=False):
        if ann["image_id"] not in transforms:
            continue
        sx, sy, pl, pt = transforms[ann["image_id"]]
        new_ann = dict(ann)
        if "bbox" in ann and ann["bbox"]:
            new_ann["bbox"] = rescale_bbox(ann["bbox"], sx, sy, pl, pt, target_w, target_h)
            new_ann["area"] = new_ann["bbox"][2] * new_ann["bbox"][3]
        if "segmentation" in ann and isinstance(ann["segmentation"], list):
            new_ann["segmentation"] = rescale_segmentation(
                ann["segmentation"], sx, sy, pl, pt, target_w, target_h)
        if "keypoints" in ann and ann["keypoints"]:
            new_ann["keypoints"] = rescale_keypoints(
                ann["keypoints"], sx, sy, pl, pt, target_w, target_h)
        new_annotations.append(new_ann)

    new_coco = dict(coco)
    new_coco["images"]      = new_images
    new_coco["annotations"] = new_annotations

    with open(out_ann_path, "w") as f:
        json.dump(new_coco, f, indent=2)

    return {"images": len(new_images), "annotations": len(new_annotations), "missing": missing}


def create_yolo_subset_dataset(src_root, dst_root, keep_percentage=0.1, seed=42):
    """
    Reduces a YOLO dataset by a percentage and copies files to a new location.
    
    Structure expected:
    src_root/
      train/
        images/
        labels/
      val/
        images/
        labels/
      test/
        images/
        labels/
    """
    random.seed(seed)
    splits = ['train', 'valid', 'test'] # Added 'valid' to match your request
    
    # Common image extensions
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for split in splits:
        # Check both 'val' and 'valid' naming conventions
        src_split_path = os.path.join(src_root, split)
        if not os.path.exists(src_split_path) and split == 'valid':
            src_split_path = os.path.join(src_root, 'val')
        
        if not os.path.exists(src_split_path):
            print(f"Skipping {split}: Directory not found.")
            continue

        src_img_dir = os.path.join(src_split_path, 'images')
        src_lbl_dir = os.path.join(src_split_path, 'labels')
        
        # Setup destination paths
        dst_img_dir = os.path.join(dst_root, split, 'images')
        dst_lbl_dir = os.path.join(dst_root, split, 'labels')
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        # List all images
        all_images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(img_extensions)]
        
        # Sample the images
        num_to_keep = int(len(all_images) * keep_percentage)
        sampled_images = random.sample(all_images, num_to_keep)

        print(f"Processing {split}: Keeping {len(sampled_images)} of {len(all_images)} images.")

        for img_name in sampled_images:
            # Copy image
            shutil.copy2(os.path.join(src_img_dir, img_name), os.path.join(dst_img_dir, img_name))
            
            # Find and copy corresponding label
            label_name = os.path.splitext(img_name)[0] + '.txt'
            src_label_path = os.path.join(src_lbl_dir, label_name)
            dst_label_path = os.path.join(dst_lbl_dir, label_name)
            
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)
            else:
                # YOLO sometimes has images without labels (background samples)
                # We create an empty file or just skip
                Path(dst_label_path).touch() 

    # Copy data.yaml if it exists
    yaml_path = os.path.join(src_root, 'data.yaml')
    if os.path.exists(yaml_path):
        shutil.copy2(yaml_path, os.path.join(dst_root, 'data.yaml'))
        print("Copied data.yaml to new root.")

def validation_dataset_split_by_match(train_path, valid_path, split_ratio=0.2):
    # Define subdirectories
    train_img_dir = Path(train_path) / "images"
    train_lbl_dir = Path(train_path) / "labels"
    valid_img_dir = Path(valid_path) / "images"
    valid_lbl_dir = Path(valid_path) / "labels"

    valid_img_dir.mkdir(parents=True, exist_ok=True)
    valid_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Group images by Match ID (e.g., SNMOT_060)
    match_groups = defaultdict(list)
    all_images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in all_images:
        parts = img_name.split('_')
        if len(parts) >= 2:
            match_id = f"{parts[0]}_{parts[1]}"
            match_groups[match_id].append(img_name)

    # 2. Select matches to move
    unique_matches = list(match_groups.keys())
    random.shuffle(unique_matches)
    
    num_matches_to_move = int(len(unique_matches) * split_ratio)
    matches_to_move = unique_matches[:num_matches_to_move]

    print(f"Total matches found: {len(unique_matches)}")
    print(f"Moving {len(matches_to_move)} matches to validation...")

    # 3. Move files
    moved_count = 0
    for match_id in matches_to_move:
        for img_name in match_groups[match_id]:
            label_name = Path(img_name).stem + ".txt"
            
            src_img = train_img_dir / img_name
            src_lbl = train_lbl_dir / label_name
            dst_img = valid_img_dir / img_name
            dst_lbl = valid_lbl_dir / label_name

            # Move Image
            if src_img.exists():
                shutil.move(str(src_img), str(dst_img))
            
            # Move Label
            if src_lbl.exists():
                shutil.move(str(src_lbl), str(dst_lbl))
            
            moved_count += 1

    print(f"Done! Moved {len(matches_to_move)} matches ({moved_count} total images).")