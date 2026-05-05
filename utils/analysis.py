
# Imports
import os
import glob
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def count_images(labels_dir: str) -> int:
    """Return the number of label files (≈ images) in *labels_dir*."""
    return len(glob.glob(os.path.join(labels_dir, "*.txt")))

def count_datasplit_samples(labels_dir: str) -> int:
    """Return the number of label files (≈ images) in *labels_dir*."""
    return len(glob.glob(os.path.join(labels_dir, "*.txt")))

def count_annotations_per_class(labels_dir: str) -> dict[int, int]:
    """
    Walk every .txt label file in *labels_dir* and tally
    annotation counts per class id.
    """
    counts: dict[int, int] = {}
    for txt_file in Path(labels_dir).glob("*.txt"):
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                counts[class_id] = counts.get(class_id, 0) + 1
    return counts

def load_class_names(yaml_path: str) -> dict[int, str]:
    """
    Parse the 'names' field from a YOLO data.yaml and return
    {class_id: class_name}.  Falls back to numeric keys if the file
    is missing or unparseable.
    """
    try:
        import yaml  # PyYAML
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        names = cfg.get('names', [])
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
    except Exception:
        pass
    return {}
 
def plot_image_counts(
    train_labels_path: str,
    val_labels_path:   str,
    test_labels_path:  str,
    figsize: tuple = (8, 4),
    save_path: str | None = None,
) -> None:
    """
    Horizontal bar chart showing the number of images in each dataset split.
 
    Parameters
    ----------
    train_labels_path : path to the training labels directory
    val_labels_path   : path to the validation labels directory
    test_labels_path  : path to the test labels directory
    figsize           : matplotlib figure size
    save_path         : if given, the figure is saved to this path
    """
    splits = ["Train", "Validation", "Test"]
    counts = [
        count_images(train_labels_path),
        count_images(val_labels_path),
        count_images(test_labels_path),
    ]
 
    # ── Styling ───────────────────────────────────────────────────────────────
    PALETTE = ["#2563EB", "#10B981", "#F59E0B"]   # blue / green / amber
    BG      = "#0F172A"
    GRID    = "#1E293B"
    TEXT    = "#F1F5F9"
 
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
 
    bars = ax.barh(splits, counts, color=PALETTE, height=0.5,
                   edgecolor="none", zorder=3)
 
    # Value labels
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va='center', ha='left',
            fontsize=11, fontweight='bold', color=TEXT,
        )
 
    # Grid & spines
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.xaxis.label.set_color(TEXT)
    ax.set_xlim(0, max(counts) * 1.18)
    ax.invert_yaxis()
 
    ax.set_xlabel("Number of Images", fontsize=11, color=TEXT, labelpad=8)
    ax.set_title("Image Count per Dataset Split", fontsize=14,
                 fontweight='bold', color=TEXT, pad=14)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"Saved → {save_path}")
    plt.show()


def plot_annotation_counts(
    labels_path: str,
    split_name:  str,
    yaml_path:   str,
    figsize:     tuple = (9, 5),
    save_path:   str | None = None,
) -> None:
    """
    Horizontal bar chart showing the number of annotations for each class
    inside a single dataset split.
 
    Parameters
    ----------
    labels_path : path to the labels directory for the desired split
                  (e.g. dataset_train_labels_path)
    split_name  : human-readable split label used in the chart title
                  (e.g. "Train", "Validation", "Test")
    yaml_path   : path to data.yaml — used to resolve class names
    figsize     : matplotlib figure size
    save_path   : if given, the figure is saved to this path
    """
    class_names = load_class_names(yaml_path)
    raw_counts  = count_annotations_per_class(labels_path)
 
    if not raw_counts:
        print(f"No annotations found in: {labels_path}")
        return
 
    # Sort by class id for a consistent order
    sorted_ids    = sorted(raw_counts.keys())
    labels        = [class_names.get(cid, f"Class {cid}") for cid in sorted_ids]
    counts        = [raw_counts[cid] for cid in sorted_ids]
 
    # ── Colour ramp ───────────────────────────────────────────────────────────
    cmap   = plt.cm.get_cmap("cool", len(labels))
    colors = [cmap(i) for i in range(len(labels))]
 
    BG   = "#0F172A"
    GRID = "#1E293B"
    TEXT = "#F1F5F9"
 
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
 
    bars = ax.barh(labels, counts, color=colors, height=0.55,
                   edgecolor="none", zorder=3)
    ax.invert_yaxis()
 
    # Value labels
    x_max = max(counts)
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + x_max * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va='center', ha='left',
            fontsize=10, fontweight='bold', color=TEXT,
        )
 
    # Grid & spines
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.xaxis.label.set_color(TEXT)
    ax.set_xlim(0, x_max * 1.18)
 
    ax.set_xlabel("Number of Annotations", fontsize=11, color=TEXT, labelpad=8)
    ax.set_title(
        f"Annotations per Class — {split_name} Split",
        fontsize=14, fontweight='bold', color=TEXT, pad=14,
    )
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"Saved → {save_path}")
    plt.show()

def plot_model_comparison_table(dataframe:pd.DataFrame, table_title:str ):
    df = dataframe.copy()
    df['model_name'] = df['model_name'].str.extract(r'\d{2}-\d{2}-\d{4}_\d{2}-\d{2}_(.+)').iloc[:, 0].str.upper()
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].map(lambda x: f"{x:.4f}")

    col_labels = ['Detection Model Name', 'Overall mAP50', 'Overall mAP50-95', 'Ball mAP50', 'Ball mAP50-95']
    cell_data  = df.values.tolist()

    n_rows = len(cell_data)
    n_cols = len(col_labels)

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.55 * n_rows))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    row_h   = 1.0 / (n_rows + 1)          # +1 for header
    col_w   = 1.0 / n_cols
    pad_x   = 0.015
    divider_color = "#e5e7eb"
    header_text_color = "#111827"
    cell_text_color   = "#374151"
    header_bg         = "white"

    # ── Draw rows ─────────────────────────────────────────────────────────────────
    for r in range(n_rows + 1):
        y_top = 1.0 - r * row_h
        y_bot = y_top - row_h
        is_header = (r == 0)

        # Alternating row background (skip header)
        if not is_header and r % 2 == 0:
            ax.add_patch(Rectangle((0, y_bot), 1, row_h, color="#f9fafb", zorder=0))

        # Horizontal divider below each row
        ax.axhline(y=y_bot, color=divider_color, linewidth=0.8, zorder=1)

        for c in range(n_cols):
            x = c * col_w + pad_x
            y_mid = (y_top + y_bot) / 2

            if is_header:
                text    = col_labels[c]
                weight  = "bold"
                color   = header_text_color
                size    = 8.5
            else:
                text    = str(cell_data[r - 1][c])
                weight  = "bold" if c == 0 else "normal"
                color   = cell_text_color
                size    = 8.5

            ax.text(x, y_mid, text,
                    va="center", ha="left",
                    fontsize=size, fontweight=weight,
                    color=color, zorder=2,
                    fontfamily="DejaVu Sans")

    # Top border
    ax.axhline(y=1.0, color=divider_color, linewidth=0.8, zorder=1)
    # Title
    fig.text(0.5, 1.02, table_title, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#111827")

    plt.tight_layout(pad=0.3)
    plt.show()


