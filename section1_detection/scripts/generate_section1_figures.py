"""Create presentation figures from the Section 1 datasets, models and notebook artifacts.

The script intentionally reads the reported evaluation values from the saved
training notebook, and uses extract_front_edge.py for the geometry definition.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib

# Generate files non-interactively; this also avoids depending on a macOS GUI backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "section1_detection" / "outputs" / "figures"
sys.path.insert(0, str(ROOT / "section1_detection" / "scripts"))
from extract_front_edge import draw_geometry, extract_front_edge  # noqa: E402


def image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Unable to read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def metrics_from_notebook() -> tuple[dict, dict]:
    """Read the two held-out result dictionaries saved by the training notebook."""
    notebook = json.loads((ROOT / "section1_detection/notebooks/01_train_section1.ipynb").read_text())
    text = "\n".join(
        "".join(output.get("text", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )
    dictionaries = []
    for match in re.finditer(r"\{'metrics/precision\(B\)'.*?'fitness': [0-9.]+\}", text):
        try:
            dictionaries.append(ast.literal_eval(match.group()))
        except (ValueError, SyntaxError):
            pass
    detection = next(d for d in dictionaries if "metrics/precision(M)" not in d)
    segmentation = next(d for d in dictionaries if "metrics/precision(M)" in d)
    return detection, segmentation


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overview() -> None:
    fig = plt.figure(figsize=(15, 9), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    fig.suptitle("Section 1 datasets: retained targets and source-group split", fontsize=20, fontweight="bold")

    ax = fig.add_subplot(grid[0, 0]); ax.axis("off")
    rows = [
        ["Dataset", "Images", "Retained annotations", "Final class"],
        ["1 · Pallet detection", "920", "10,329 pallet", "pallet"],
        ["2 · Pallet geometry", "412", "278 pallet_front", "pallet_front"],
    ]
    table = ax.table(cellText=rows, cellLoc="left", loc="center", colWidths=[.36, .15, .28, .21])
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 2.2)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#17324d" if r == 0 else ("#eaf2f8" if r == 1 else "#f9f3e8"))
        if r == 0: cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
    ax.set_title("Documented dataset totals", loc="left", fontweight="bold", pad=12)

    ax = fig.add_subplot(grid[0, 1]); ax.axis("off")
    definitions = (
        "Dataset 1 source classes: hole, pallet\n"
        "Final detection target: pallet only\n\n"
        "Dataset 2 source classes: front, hole, hole_left, hole_right,\n"
        "pallet, pallet_front, pallet_pocket, wood\n"
        "Final geometry target: pallet_front only"
    )
    ax.add_patch(FancyBboxPatch((.02, .10), .96, .78, boxstyle="round,pad=.025", facecolor="#f7f9fb", edgecolor="#b8c7d3"))
    ax.text(.06, .78, definitions, va="top", fontsize=13, linespacing=1.5)
    ax.set_title("Class definitions", loc="left", fontweight="bold", pad=12)

    ax = fig.add_subplot(grid[1, :])
    datasets = {
        "Dataset 1\n(images)": [644, 141, 135],
        "Dataset 1\n(instances)": [6929, 1527, 1873],
        "Dataset 2\n(images)": [288, 61, 63],
        "Dataset 2\n(instances)": [199, 42, 37],
    }
    labels = ["Train", "Validation", "Held-out test"]
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    x = np.arange(len(datasets)); bottom = np.zeros(len(datasets))
    for idx, label in enumerate(labels):
        vals = np.array([v[idx] for v in datasets.values()])
        ax.bar(x, vals, bottom=bottom, color=colors[idx], label=label, width=.62)
        for xi, val, b in zip(x, vals, bottom): ax.text(xi, b + val / 2, f"{val:,}", ha="center", va="center", fontsize=10, fontweight="bold")
        bottom += vals
    ax.set_xticks(x, datasets.keys(), fontsize=11); ax.set_ylabel("Count (split stacked)")
    ax.legend(ncol=3, loc="upper center"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Final split counts", loc="left", fontweight="bold")
    ax.text(.01, -.27, "Deterministic source-group-aware split (seed 42): 70% train / 15% validation / 15% held-out.  "
            "Dataset 1: 378 groups (264 / 56 / 58).  Dataset 2: 412 derived groups (288 / 61 / 63).",
            transform=ax.transAxes, fontsize=11)
    save(fig, "dataset_overview.png")


def predicted_mask(result, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if result.masks is None or result.masks.data is None:
        return mask
    masks = result.masks.data.cpu().numpy()
    for item in masks:
        item = cv2.resize(item.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        mask |= (item > .5).astype(np.uint8)
    return mask


def inference_examples() -> tuple[list, list, YOLO]:
    det_paths = sorted((ROOT / "data/processed/detection/images/test").glob("*"))[:3]
    seg_dir = ROOT / "data/processed/geometry"
    seg_paths = []
    for label in sorted((seg_dir / "labels/test").glob("*.txt")):
        if label.read_text().strip():
            candidate = seg_dir / "images/test" / f"{label.stem}.jpg"
            if candidate.exists(): seg_paths.append(candidate)
        if len(seg_paths) == 3: break
    detector = YOLO(str(ROOT / "section1_detection/models/pallet_detector_best.pt"))
    segmenter = YOLO(str(ROOT / "section1_detection/models/pallet_geometry_best.pt"))
    det_results = list(detector.predict([str(p) for p in det_paths], conf=.25, verbose=False, device="cpu"))
    seg_results = list(segmenter.predict([str(p) for p in seg_paths], conf=.25, verbose=False, device="cpu"))
    return list(zip(det_paths, det_results)), list(zip(seg_paths, seg_results)), segmenter


def detection_examples(samples: list) -> None:
    fig, axes = plt.subplots(1, len(samples), figsize=(16, 5.4), layout="constrained")
    fig.suptitle("Held-out pallet detection examples", fontsize=18, fontweight="bold")
    for ax, (path, result) in zip(axes, samples):
        rendered = cv2.cvtColor(result.plot(labels=True, conf=True), cv2.COLOR_BGR2RGB)
        ax.imshow(rendered); ax.set_title(path.name, fontsize=8); ax.axis("off")
    fig.text(.5, .02, "Blue boxes: predicted pallet detections; labels show model confidence. Images are from the held-out split.", ha="center", fontsize=10)
    save(fig, "detection_examples.png")


def segmentation_examples(samples: list) -> None:
    fig, axes = plt.subplots(len(samples), 3, figsize=(13, 12), layout="constrained")
    fig.suptitle("Held-out pallet-front segmentation examples", fontsize=18, fontweight="bold")
    for r, (path, result) in enumerate(samples):
        img = image_rgb(path); mask = predicted_mask(result, img.shape[:2])
        overlay = img.copy(); overlay[mask > 0] = (0.45 * overlay[mask > 0] + 0.55 * np.array([34, 197, 94])).astype(np.uint8)
        axes[r, 0].imshow(img); axes[r, 0].set_title(f"Original\n{path.name}", fontsize=8)
        axes[r, 1].imshow(mask, cmap="Greens", vmin=0, vmax=1); axes[r, 1].set_title("Predicted pallet-front mask")
        axes[r, 2].imshow(overlay); axes[r, 2].set_title("Predicted region overlay")
        for ax in axes[r]: ax.axis("off")
    save(fig, "segmentation_examples.png")


def front_edge_geometry(samples: list) -> None:
    path, result = samples[0]
    img = image_rgb(path); mask = predicted_mask(result, img.shape[:2])
    g0, g1 = extract_front_edge(mask)
    geometry = draw_geometry(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), g0, g1)
    geometry = cv2.cvtColor(geometry, cv2.COLOR_BGR2RGB)
    length = float(np.hypot(g1[0] - g0[0], g1[1] - g0[1])) if g0 and g1 else None
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
    fig.suptitle("Pallet-front geometry extracted from predicted segmentation", fontsize=18, fontweight="bold")
    axes[0].imshow(img); axes[0].set_title(f"Input image\n{path.name}", fontsize=9)
    axes[1].imshow(mask, cmap="Greens", vmin=0, vmax=1); axes[1].set_title("Predicted pallet-front mask")
    axes[2].imshow(geometry); axes[2].set_title("G0 → G1 front edge")
    if g0 and g1:
        axes[2].text(g0[0], g0[1] - 18, "G0", color="lime", fontsize=12, fontweight="bold")
        axes[2].text(g1[0], g1[1] - 18, "G1", color="red", fontsize=12, fontweight="bold")
        axes[2].text(.02, .04, f"pixel edge length: {length:.1f} px", transform=axes[2].transAxes,
                     color="white", fontsize=11, bbox=dict(facecolor="black", alpha=.65, pad=4))
    for ax in axes: ax.axis("off")
    fig.text(.5, .01, "G0 and G1 use Section 1's extract_front_edge.py: left/right extrema of the mask's bottom 10%.", ha="center", fontsize=10)
    save(fig, "front_edge_geometry.png")


def evaluation_summary() -> None:
    det, seg = metrics_from_notebook()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    fig.suptitle("Section 1 held-out evaluation summary", fontsize=19, fontweight="bold")
    names = ["Precision", "Recall", "mAP50", "mAP50–95"]
    det_vals = [det["metrics/precision(B)"], det["metrics/recall(B)"], det["metrics/mAP50(B)"], det["metrics/mAP50-95(B)"]]
    box_vals = [seg["metrics/precision(B)"], seg["metrics/recall(B)"], seg["metrics/mAP50(B)"], seg["metrics/mAP50-95(B)"]]
    mask_vals = [seg["metrics/precision(M)"], seg["metrics/recall(M)"], seg["metrics/mAP50(M)"], seg["metrics/mAP50-95(M)"]]
    x = np.arange(4); width = .24
    for offset, vals, label, color in [(-width, det_vals, "Detector boxes", "#2878b5"), (0, box_vals, "Geometry boxes", "#e89b38"), (width, mask_vals, "Geometry masks", "#35a675")]:
        bars = axes[0].bar(x + offset, vals, width, label=label, color=color)
        axes[0].bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=2, fontsize=8, rotation=90)
    axes[0].set_ylim(0, 1.12); axes[0].set_xticks(x, names); axes[0].set_ylabel("Score"); axes[0].legend(loc="lower left"); axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Metrics read from saved training notebook", loc="left", fontweight="bold")
    axes[1].axis("off")
    rows = [
        ["Evaluation set", "Images", "Instances"],
        ["Pallet detection", "135", "1,873 pallet"],
        ["Pallet-front segmentation", "63", "37 pallet_front"],
        ["", "", ""],
        ["Reported held-out result", "", ""],
        ["Detector mAP50–95", "", f"{det_vals[-1]:.3f}"],
        ["Segmentation mask mAP50–95", "", f"{mask_vals[-1]:.3f}"],
    ]
    table = axes[1].table(cellText=rows, loc="center", cellLoc="left", colWidths=[.56, .19, .25])
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.05, 2.15)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r in (0, 4): cell.set_facecolor("#17324d"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        elif r != 3: cell.set_facecolor("#eef4f8")
    axes[1].set_title("Held-out split context", loc="left", fontweight="bold", pad=12)
    save(fig, "evaluation_summary.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    overview()
    det_samples, seg_samples, _ = inference_examples()
    detection_examples(det_samples)
    segmentation_examples(seg_samples)
    front_edge_geometry(seg_samples)
    evaluation_summary()
    print("Generated:")
    for path in sorted(OUT.glob("*.png")): print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
