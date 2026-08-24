"""Generate report figures from saved Section 3 and Section 2 artifacts only.

This script does not run inference or change SOP logic.  In particular, empty
YOLOE detections remain empty and MANUAL_INSPECTION is never recast as PASS/FAIL.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "section3_implementation" / "outputs" / "figures"


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_card(ax, x: float, y: float, w: float, h: float, title: str, body: str, color: str = "#eef4f8") -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.018", facecolor=color, edgecolor="#627d98", linewidth=1.3))
    ax.text(x + .02, y + h - .055, title, va="top", fontsize=11, weight="bold")
    ax.text(x + .02, y + h - .11, body, va="top", fontsize=9.5, wrap=True, linespacing=1.35)


def input_figure(image: np.ndarray, assessment: dict, pose_artifact: dict) -> None:
    pose = assessment["pose"]
    quality = assessment["pose_quality"]
    s2 = pose_artifact["assessments"][0]
    fig = plt.figure(figsize=(15, 8), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
    fig.suptitle("Section 3 input: saved pallet image, pose interface, and load evidence", fontsize=19, fontweight="bold")
    ax = fig.add_subplot(grid[0, 0]); ax.imshow(image); ax.axis("off"); ax.set_title("Saved Section 3 input image: section3_image_sample.jpg", fontsize=10)
    ax = fig.add_subplot(grid[0, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    add_card(ax, .04, .68, .92, .23, "Section 2 pose interface", f"x={pose['x_m']:.3f} m   y={pose['y_m']:.3f} m   θ={pose['theta_deg']:.1f}°\n"
             f"status: {quality['status']}   geometry confidence: {quality['geometry_confidence']:.2f}", "#fdecea")
    add_card(ax, .04, .42, .92, .20, "Section 1/2 front geometry", f"G0={tuple(s2['G0_px'])} px   G1={tuple(s2['G1_px'])} px\n"
             f"reprojection RMS={s2['reprojection_rms_px']:.2f} px (saved Section 2 artifact)", "#fff4df")
    detections = assessment["detections"]
    add_card(ax, .04, .18, .92, .18, "YOLOE-Seg load evidence", f"Saved Section 3 output: {len(detections)} detected load/box instances.\n"
             "No detected boxes or masks are available to render.", "#f3f4f6")
    ax.text(.5, .07, "Metric SOP checks are constrained because the incoming Section 2 pose is UNRELIABLE.", ha="center", color="#b23a48", fontsize=10.5, weight="bold", wrap=True)
    save(fig, "section3_input.png")


def detections_figure(image: np.ndarray, assessment: dict) -> None:
    detections = assessment["detections"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    fig.suptitle("YOLOE-Seg load detection evidence", fontsize=19, fontweight="bold")
    axes[0].imshow(image); axes[0].axis("off"); axes[0].set_title("Saved input image")
    axes[1].axis("off"); axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    if detections:
        # This branch preserves existing saved detections verbatim if a future artifact includes them.
        axes[1].text(.05, .92, f"Saved detections: {len(detections)}", fontsize=14, weight="bold")
        for idx, det in enumerate(detections[:8]):
            axes[1].text(.07, .83 - idx*.09, f"{det.get('label', 'box')}  confidence={det.get('confidence', 0):.2f}", fontsize=11)
    else:
        add_card(axes[1], .06, .59, .88, .22, "YOLOE-Seg result", "0 load/box instances in the saved Section 3 output.", "#fdecea")
        add_card(axes[1], .06, .31, .88, .20, "Boxes, masks, and confidence", "Unavailable because no detections were returned. No synthetic detection overlays have been added.", "#f3f4f6")
        add_card(axes[1], .06, .09, .88, .15, "Configured prompts", "cardboard box · carton · package · box", "#eef4f8")
    save(fig, "load_detections.png")


def evidence_figure(image: np.ndarray, assessment: dict) -> None:
    fig = plt.figure(figsize=(15, 8), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
    fig.suptitle("SOP evidence visualization: available versus unavailable evidence", fontsize=19, fontweight="bold")
    ax = fig.add_subplot(grid[0, 0]); ax.imshow(image, alpha=.72); ax.axis("off")
    ax.text(.5, .50, "NO COMPUTABLE LOAD GEOMETRY", transform=ax.transAxes, ha="center", va="center", fontsize=18, weight="bold", color="#8b1e3f", bbox=dict(facecolor="white", alpha=.88, boxstyle="round,pad=.55"))
    ax.text(.5, .42, "No YOLOE-Seg load detections; metric pallet boundary unavailable because pose status is UNRELIABLE.", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#45202a", bbox=dict(facecolor="white", alpha=.88, boxstyle="round,pad=.4"))
    ax = fig.add_subplot(grid[0, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    add_card(ax, .04, .72, .92, .18, "Pallet boundary / overhang evidence", "Unavailable: no valid metric pallet frame. Rule 1 remains MANUAL_INSPECTION.", "#fdecea")
    add_card(ax, .04, .49, .92, .17, "Visible load geometry / box orientation", "Unavailable: zero detected boxes or instance masks. Rule 3 remains MANUAL_INSPECTION.", "#f3f4f6")
    add_card(ax, .04, .26, .92, .17, "Visible centroid", "Unavailable: no detected load geometry and no reliable camera metric interface. Rule 7 remains MANUAL_INSPECTION.", "#f3f4f6")
    add_card(ax, .04, .06, .92, .13, "Conservative policy", "Unavailable or partial one-view evidence is not converted into PASS or FAIL.", "#fff4df")
    save(fig, "sop_evidence.png")


def rules_figure(assessment: dict) -> None:
    checks = assessment["sop_checks"]
    fig, ax = plt.subplots(figsize=(17, 9), layout="constrained")
    fig.suptitle("Rule-level SOP compliance summary", fontsize=19, fontweight="bold")
    ax.axis("off")
    rows = [[f"R{r['rule_id']}", r["name"], r["verdict"], f"{r['confidence']:.2f}", r["observability"], textwrap.fill(r["reason"], width=55)] for r in checks]
    table = ax.table(cellText=rows, colLabels=["Rule", "Check", "Verdict", "Confidence", "Observability", "Reason"], cellLoc="left", colLoc="left", loc="center", colWidths=[.06, .15, .18, .10, .13, .38])
    table.auto_set_font_size(False); table.set_fontsize(9.3); table.scale(1, 2.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor("#17324d"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#fff4df" if c == 2 else "#eef4f8")
            if c == 2: cell.get_text().set_weight("bold")
    fig.text(.5, .025, "All outcomes are reproduced from the saved Section 3 assessment. MANUAL_INSPECTION denotes insufficient or partially observable evidence, not compliance.", ha="center", fontsize=10)
    save(fig, "rule_compliance_summary.png")


def summary_figure(assessment: dict, pose_artifact: dict) -> None:
    checks = assessment["sop_checks"]
    verdicts = {v: sum(r["verdict"] == v for r in checks) for v in ("PASS", "FAIL", "MANUAL_INSPECTION")}
    s2 = pose_artifact["assessments"][0]
    fig = plt.figure(figsize=(15, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.25])
    fig.suptitle("Final Section 3 assessment summary", fontsize=20, fontweight="bold")
    ax = fig.add_subplot(grid[:, 0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    overall = assessment["overall_verdict"]
    add_card(ax, .06, .72, .88, .19, "Overall verdict", overall, "#fff4df")
    add_card(ax, .06, .49, .88, .17, "Pose quality", f"Section 2 status: {assessment['pose_quality']['status']}\n"
             f"Saved reprojection RMS: {s2['reprojection_rms_px']:.2f} px", "#fdecea")
    add_card(ax, .06, .28, .88, .15, "Detected load/box instances", str(len(assessment["detections"])), "#f3f4f6")
    add_card(ax, .06, .07, .88, .15, "Rule outcomes", f"PASS: {verdicts['PASS']}   FAIL: {verdicts['FAIL']}   MANUAL_INSPECTION: {verdicts['MANUAL_INSPECTION']}", "#eef4f8")
    ax = fig.add_subplot(grid[0, 1]); ax.axis("off")
    ax.text(0, 1, "Why metric compliance is not established", va="top", fontsize=14, weight="bold")
    reasons = assessment["overall_reason"]
    ax.text(.02, .82, "\n\n".join(f"• {reason}" for reason in reasons[:4]), va="top", fontsize=10, wrap=True, linespacing=1.35)
    ax = fig.add_subplot(grid[1, 1]); ax.axis("off")
    ax.text(0, 1, "Evidence interpretation", va="top", fontsize=14, weight="bold")
    ax.text(.02, .79, "Computable from this saved run\n• Rule outcomes, observation states, and configured confidence values\n• Section 2 candidate pose and its UNRELIABLE quality status\n\nUnavailable / insufficient\n• YOLOE load detections, masks, box orientation, load centroid, and metric overhang evidence\n• Physical ground truth or a reliable metric pose", va="top", fontsize=10.5, linespacing=1.45)
    save(fig, "assessment_summary.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assessment = load_json("section3_output.json")
    pose_artifact = load_json("section2_pose/artifacts/real_pose/pose_assessment.json")
    image = plt.imread(ROOT / "section3_image_sample.jpg")
    input_figure(image, assessment, pose_artifact)
    detections_figure(image, assessment)
    evidence_figure(image, assessment)
    rules_figure(assessment)
    summary_figure(assessment, pose_artifact)
    print("Generated:")
    for figure in sorted(OUT.glob("*.png")):
        print(figure.relative_to(ROOT))


if __name__ == "__main__":
    main()
