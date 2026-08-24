"""Create concise Section 2 submission figures from existing artifacts and models.

No Section 2 estimation logic is reimplemented here.  The only live inference
uses the existing Section1GeometryAdapter to illustrate its fixed G0/G1 input
interface; all reported evaluation values are read from saved JSON artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "section2_pose" / "outputs" / "figures"
sys.path.insert(0, str(ROOT))

from section2_pose.inference.section1_adapter import Section1GeometryAdapter  # noqa: E402


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def example_geometry() -> tuple[Path, np.ndarray, dict]:
    """Get one representative existing test image through the frozen adapter."""
    adapter = Section1GeometryAdapter(device="cpu")
    candidates = sorted((ROOT / "data/processed/geometry/images/test").glob("*"))
    for path in candidates:
        image = cv2.imread(str(path))
        if image is None:
            continue
        detections = adapter.predict(image)
        if detections and detections[0]["G0"] is not None and detections[0]["G1"] is not None:
            return path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections[0]
    raise RuntimeError("No usable Section 1 pallet-front geometry found in the existing geometry test split.")


def geometry_overlay(image: np.ndarray, item: dict, *, labels: bool = True) -> np.ndarray:
    output = image.copy()
    mask = item["mask"].astype(bool)
    output[mask] = (0.42 * output[mask] + 0.58 * np.array([24, 160, 88])).astype(np.uint8)
    g0, g1 = item["G0"], item["G1"]
    cv2.circle(output, g0, 8, (36, 255, 24), -1)
    cv2.circle(output, g1, 8, (245, 64, 55), -1)
    cv2.line(output, g0, g1, (35, 91, 245), 4)
    if labels:
        cv2.putText(output, "G0", (g0[0] - 32, g0[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, "G1", (g1[0] + 12, g1[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 70, 50), 2, cv2.LINE_AA)
    return output


def input_geometry_figure(path: Path, image: np.ndarray, item: dict) -> None:
    overlay = geometry_overlay(image, item)
    confidence = item["confidence"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    fig.suptitle("Section 2 input: pallet image and Section 1 front geometry", fontsize=18, fontweight="bold")
    axes[0].imshow(image); axes[0].set_title(f"Geometry-test image\n{path.name}", fontsize=9)
    axes[1].imshow(overlay)
    axes[1].set_title(f"pallet_front mask + G0/G1  |  confidence {confidence:.2f}" if confidence is not None else "pallet_front mask + G0/G1")
    for ax in axes: ax.axis("off")
    fig.text(.5, .02, "Green: predicted pallet-front region · blue: visible lower/front edge · endpoints are the fixed Section 1 → Section 2 interface.", ha="center", fontsize=10)
    save(fig, "input_front_geometry.png")


def edge_figure(path: Path, image: np.ndarray, item: dict) -> None:
    mask = item["mask"].astype(np.uint8)
    edge = geometry_overlay(image, item)
    g0, g1 = item["G0"], item["G1"]
    length = float(np.linalg.norm(np.asarray(g1) - np.asarray(g0)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
    fig.suptitle("G0 → G1 lower/front-edge extraction", fontsize=18, fontweight="bold")
    axes[0].imshow(image); axes[0].set_title("Input image")
    axes[1].imshow(mask, cmap="Greens", vmin=0, vmax=1); axes[1].set_title("Section 1 predicted pallet_front mask")
    axes[2].imshow(edge); axes[2].set_title(f"Extracted front edge: {length:.1f} px")
    for ax in axes: ax.axis("off")
    fig.text(.5, .02, "Existing Section 1 extraction: take the mask's bottom 10%, then use its image-left (G0) and image-right (G1) extrema.", ha="center", fontsize=10)
    save(fig, "front_edge_extraction.png")


def pose_frame_figure(assessment: dict) -> None:
    item = assessment["assessments"][0]
    pose = item["pose"]
    pallet = assessment["pallet_geometry"]
    center = np.array([pose["x_m"], pose["y_m"]])
    theta = np.deg2rad(pose["theta_deg"])
    x_axis = np.array([np.cos(theta), np.sin(theta)])
    y_axis = np.array([-np.sin(theta), np.cos(theta)])
    length, width = pallet["length_m"], pallet["width_m"]
    corners = np.array([center + sx * length / 2 * x_axis + sy * width / 2 * y_axis for sx, sy in [(1,1),(1,-1),(-1,-1),(-1,1)]])
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    fig.suptitle("Real-image integration: candidate floor-frame pose", fontsize=18, fontweight="bold")
    ax.add_patch(Polygon(corners, closed=True, facecolor="#9fc5e8", edgecolor="#175a91", linewidth=2.5, label="Nominal pallet footprint"))
    front_mid = center + length / 2 * x_axis
    ax.plot([corners[0,0], corners[1,0]], [corners[0,1], corners[1,1]], color="#e76f51", lw=5, label="Visible front edge")
    ax.scatter(*center, s=65, color="#17324d", zorder=4)
    ax.add_patch(FancyArrowPatch(center, center + .65 * x_axis, arrowstyle="->", mutation_scale=18, color="#d1495b", linewidth=2.5))
    ax.add_patch(FancyArrowPatch(center, center + .55 * y_axis, arrowstyle="->", mutation_scale=18, color="#2a9d8f", linewidth=2.5))
    ax.text(*(center + .70 * x_axis), "+X (pallet front)", color="#d1495b", fontsize=11, weight="bold")
    ax.text(*(center + .60 * y_axis), "+Y (pallet left)", color="#2a9d8f", fontsize=11, weight="bold")
    ax.scatter(0, 0, marker="+", s=130, linewidths=2.5, color="black")
    ax.text(.03, .03, "floor origin\n(camera-floor projection)", transform=ax.transAxes, fontsize=10)
    ax.set_aspect("equal"); ax.grid(alpha=.25); ax.set_xlabel("floor X (m)"); ax.set_ylabel("floor Y (m)")
    ax.set_xlim(-.5, 3.2); ax.set_ylim(-1.6, 1.1); ax.legend(loc="upper right")
    status = item["pose_status"].replace("_", " ")
    ax.text(.02, .98, f"Candidate: x={pose['x_m']:.3f} m, y={pose['y_m']:.3f} m, θ={pose['theta_deg']:.1f}°\n"
                        f"Status: {status}\nReason: {item['failure_reason']}\n"
                        f"Reprojection RMS: {item['reprojection_rms_px']:.2f} px (threshold: 3.00 px)",
            transform=ax.transAxes, va="top", fontsize=10, bbox=dict(facecolor="#fff4e6", edgecolor="#e76f51", boxstyle="round,pad=.5"))
    fig.text(.5, .01, "This is the saved real-image integration artifact. It is visualized as a candidate only; the artifact marks it UNRELIABLE and does not permit metric use.", ha="center", fontsize=9.5)
    save(fig, "estimated_pose_frame.png")


def pipeline_figure(assessment: dict) -> None:
    fig, ax = plt.subplots(figsize=(16, 6), layout="constrained")
    fig.suptitle("Section 2 pose-estimation geometry transformation", fontsize=18, fontweight="bold")
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    entries = [
        ("Section 1 segmentation", "pallet_front mask", "#e8f1fa"),
        ("Front-edge interface", "G0, G1 in image pixels", "#eaf7f0"),
        ("Ray / floor intersection", "camera intrinsics + height + tilt", "#fff4df"),
        ("Metric edge geometry", "midpoint + known 1.00 m edge", "#f6ecfa"),
        ("Constrained refinement", "reproject nominal 3-D G0/G1", "#e8f1fa"),
        ("Output assessment", "x, y, θ + residual / status", "#fdecea"),
    ]
    xs = np.linspace(.09, .91, len(entries))
    for idx, ((title, detail, color), x) in enumerate(zip(entries, xs)):
        ax.add_patch(FancyBboxPatch((x-.07, .42), .14, .23, boxstyle="round,pad=.015", facecolor=color, edgecolor="#46637f", linewidth=1.5))
        ax.text(x, .57, title, ha="center", va="center", weight="bold", fontsize=10, wrap=True)
        ax.text(x, .47, detail, ha="center", va="center", fontsize=8, wrap=True)
        if idx < len(entries)-1: ax.add_patch(FancyArrowPatch((x+.075,.535), (xs[idx+1]-.075,.535), arrowstyle="->", mutation_scale=16, linewidth=1.8, color="#46637f"))
    cam = assessment["calibration"]; pal = assessment["pallet_geometry"]
    ax.text(.5, .22, f"Reference camera assumptions: {cam['image_size_px'][0]}×{cam['image_size_px'][1]} px, fx=fy={cam['fx_px']:.0f} px, height={cam['height_m']:.2f} m, tilt={cam['tilt_deg']:.0f}°.  "
                      f"Nominal pallet: {pal['length_m']:.2f} × {pal['width_m']:.2f} × {pal['deck_height_m']:.3f} m.", ha="center", fontsize=10)
    ax.text(.5, .12, "G0 is image-left and G1 image-right. On the floor plane, their midpoint locates the front edge; the right-handed perpendicular gives pallet +X/yaw.", ha="center", fontsize=10)
    save(fig, "pose_geometry_pipeline.png")


def evaluation_figure(calibration: dict, evaluation: dict, assessment: dict) -> None:
    noise = evaluation["landmark_noise"]
    sigmas = np.array([float(k) for k in noise])
    trans_p95 = [noise[str(s)]["translation_m"]["p95"] for s in sigmas]
    rot_p95 = [noise[str(s)]["rotation_deg"]["p95"] for s in sigmas]
    pass_rates = [noise[str(s)]["pass_rate"] for s in sigmas]
    fig = plt.figure(figsize=(16, 9), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=[1.3, 1])
    fig.suptitle("Section 2 evaluation summary", fontsize=20, fontweight="bold")
    ax = fig.add_subplot(grid[0, 0])
    ax.plot(sigmas, trans_p95, "o-", color="#2878b5", label="P95 translation error (m)")
    ax.axhline(evaluation["acceptance"]["translation_m"], color="#d1495b", linestyle="--", label="2 cm acceptance")
    ax.set_xlabel("Endpoint noise σ (px)"); ax.set_ylabel("P95 translation error (m)"); ax.grid(alpha=.25); ax.legend(fontsize=9); ax.set_title("Synthetic/reference landmark-noise evaluation", loc="left", fontweight="bold")
    ax = fig.add_subplot(grid[1, 0])
    ax.plot(sigmas, rot_p95, "o-", color="#e89b38", label="P95 rotation error (deg)")
    ax.axhline(evaluation["acceptance"]["rotation_deg"], color="#d1495b", linestyle="--", label="3° acceptance")
    ax.set_xlabel("Endpoint noise σ (px)"); ax.set_ylabel("P95 rotation error (deg)"); ax.grid(alpha=.25); ax.legend(fontsize=9)
    ax = fig.add_subplot(grid[2, 0])
    ax.plot(sigmas, pass_rates, "s--", color="#2a9d8f", label="Pass rate")
    ax.axhline(evaluation["acceptance"]["operating_envelope_pass_rate"], color="#d1495b", linestyle="--", label="95% acceptance")
    ax.set_ylim(-.05, 1.08); ax.set_xlabel("Endpoint noise σ (px)"); ax.set_ylabel("Pass rate (fraction)"); ax.grid(alpha=.25); ax.legend(fontsize=9)
    ax = fig.add_subplot(grid[:, 1]); ax.axis("off")
    real = assessment["assessments"][0]
    rows = [
        ["Artifact / result", "Value"],
        ["Synthetic calibration RMS", f"{calibration['calibration_rms_px']:.3f} px"],
        ["Calibration P95 / max", f"{calibration['per_point_reprojection_error_px']['p95']:.3f} / {calibration['per_point_reprojection_error_px']['max']:.3f} px"],
        ["Zero-noise synthetic pass rate", f"{evaluation['zero_noise']['pass_rate']:.0%}"],
        ["1 px noise: P95 translation", f"{noise['1.0']['translation_m']['p95']:.3f} m"],
        ["1 px noise: P95 rotation", f"{noise['1.0']['rotation_deg']['p95']:.3f}°"],
        ["1 px noise: pass rate", f"{noise['1.0']['pass_rate']:.0%}"],
        ["Real-image geometry", f"G0={tuple(real['G0_px'])}, G1={tuple(real['G1_px'])}"],
        ["Real-image reprojection RMS", f"{real['reprojection_rms_px']:.2f} px"],
        ["Real-image status", real['pose_status']],
    ]
    table = ax.table(cellText=rows, cellLoc="left", colWidths=[.60, .40], loc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.05, 2.0)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0: cell.set_facecolor("#17324d"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        elif r == len(rows)-1: cell.set_facecolor("#fdecea"); cell.get_text().set_weight("bold")
        else: cell.set_facecolor("#eef4f8")
    ax.set_title("Saved artifact values\n(synthetic/reference except marked real-image row)", loc="left", fontweight="bold", pad=15)
    save(fig, "evaluation_summary.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assessment = load_json("section2_pose/artifacts/real_pose/pose_assessment.json")
    calibration = load_json("section2_pose/artifacts/calibration/synthetic_calibration.json")
    evaluation = load_json("section2_pose/artifacts/evaluation/pose_evaluation_summary.json")
    path, image, item = example_geometry()
    input_geometry_figure(path, image, item)
    edge_figure(path, image, item)
    pose_frame_figure(assessment)
    pipeline_figure(assessment)
    evaluation_figure(calibration, evaluation, assessment)
    print("Generated:")
    for figure in sorted(OUT.glob("*.png")):
        print(figure.relative_to(ROOT))


if __name__ == "__main__":
    main()
