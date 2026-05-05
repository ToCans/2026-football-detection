# Imports
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def gather_least_certain_avg(jpg_files, dataset_images_directory, detection_model, upper_uncertainty, top_n, grab_label = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    uncertainty_records = []

    for root, file in tqdm(jpg_files, desc="Scoring images", unit="img"):
        # Gathering Correct Paths
        image_file_name = file[:-4]
        full_image_path = os.path.join(dataset_images_directory, f"{image_file_name}.jpg")

        # Ensure no training is performed
        detection_model.eval()
        detection_results = detection_model.predict(full_image_path, imgsz=1280, conf=0.1, verbose=False)

        # ── Compute per-class uncertainty ──────────────────────────────────────
        boxes = detection_results[0].boxes
        class_names = detection_model.names  # {0: 'ball', 1: 'goalkeeper', ...}
        per_class_unc = {name: [] for name in class_names.values()}

        if boxes is not None and len(boxes) > 0:
            confs   = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for conf, cls_id in zip(confs, cls_ids):
                class_name = class_names[cls_id]
                per_class_unc[class_name].append(1.0 - float(conf))  # least-confidence

        # Aggregate: mean uncertainty per class (None if no detections for that class)
        record: dict[str, float | str | None] = {"image": full_image_path}

        if grab_label:
            dataset_labels_directory = dataset_images_directory.replace("images", "labels")
            full_label_path = os.path.join(dataset_labels_directory, f"{image_file_name}.txt")
            record["label"] = full_label_path

        overall_uncs = []
        for class_name, uncs in per_class_unc.items():
            if uncs:
                mean_unc = float(np.mean(uncs))
                record[f"unc_{class_name}"] = round(mean_unc, 6)
                overall_uncs.append(mean_unc)
            else:
                record[f"unc_{class_name}"] = None

        record["unc_mean"] = round(float(np.mean(overall_uncs)), 6) if overall_uncs else None
        uncertainty_records.append(record)

    # ── Build dataframe ────────────────────────────────────────────────────────
    uncertainty_df = pd.DataFrame(uncertainty_records)

    # ── Apply confidence band filter ───────────────────────────────────────────
    # Keep only images where the overall mean uncertainty falls within [lower, upper]
    uncertainty_df = uncertainty_df[
        uncertainty_df["unc_mean"].notna() &
        (uncertainty_df["unc_mean"] <= upper_uncertainty)
    ].copy()

    # ── Sort by highest uncertainty first (best candidates to annotate) ─────────
    uncertainty_df.sort_values("unc_mean", ascending=True, inplace=True)
    uncertainty_df.reset_index(drop=True, inplace=True)

    # ── Select top N ───────────────────────────────────────────────────────────
    top_df = uncertainty_df.head(top_n).copy()
    top_df["selected_rank"] = range(1, len(top_df) + 1)

    print(f"Top {len(top_df)} images selected for annotation.")
    return uncertainty_df, top_df
    
def gather_least_certain_ball(jpg_files, dataset_images_directory, detection_model, lower_uncertainty, top_n, grab_label = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    uncertainty_records = []
    for root, file in tqdm(jpg_files, desc="Scoring images", unit="img"):
        image_file_name = file[:-4]
        full_image_path = os.path.join(dataset_images_directory, f"{image_file_name}.jpg")
        detection_model.eval()
        detection_results = detection_model.predict(full_image_path, imgsz=1280, conf=lower_uncertainty, verbose=False)

        # ── Compute per-class uncertainty ──────────────────────────────────
        boxes = detection_results[0].boxes
        class_names = detection_model.names
        per_class_unc = {name: [] for name in class_names.values()}
        if boxes is not None and len(boxes) > 0:
            confs   = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for conf, cls_id in zip(confs, cls_ids):
                class_name = class_names[cls_id]
                per_class_unc[class_name].append(1.0 - float(conf))

        record: dict[str, float | str | None] = {"image": full_image_path}

        if grab_label:
            dataset_labels_directory = dataset_images_directory.replace("images", "labels")
            full_label_path = os.path.join(dataset_labels_directory, f"{image_file_name}.txt")
            record["label"] = full_label_path

        overall_uncs = []
        for class_name, uncs in per_class_unc.items():
            if uncs:
                mean_unc = float(np.mean(uncs))
                record[f"unc_{class_name}"] = round(mean_unc, 6)
                overall_uncs.append(mean_unc)
            else:
                record[f"unc_{class_name}"] = None

        record["unc_mean"] = round(float(np.mean(overall_uncs)), 6) if overall_uncs else None

        # ── Ball-specific flags ─────────────────────────────────────────────
        ball_unc = record.get("unc_ball")
        record["ball_not_detected"] = ball_unc is None

        uncertainty_records.append(record)

    # ── Build dataframe ────────────────────────────────────────────────────
    uncertainty_df = pd.DataFrame(uncertainty_records)
    if "unc_ball" not in uncertainty_df.columns:
        uncertainty_df["unc_ball"] = None

    # ── Sort: no-detection first, then least certain ───────────────────────
    uncertainty_df.sort_values(
        ["ball_not_detected", "unc_ball"],
        ascending=[False, False],
        inplace=True,
        na_position="last"
    )
    uncertainty_df.reset_index(drop=True, inplace=True)

    # ── Select top N ──────────────────────────────────────────────────────
    top_df = uncertainty_df.head(top_n).copy()
    top_df["selected_rank"] = range(1, len(top_df) + 1)
    print(f"Top {len(top_df)} images selected for annotation "
          f"({top_df['ball_not_detected'].sum()} with no ball detected).")
    return uncertainty_df, top_df