
#!/usr/bin/env python3
"""Generate report-ready Section 4 figures from measured artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.4, 4.2, 2.0, 0.9, "Camera / Frame"),
        (3.0, 4.2, 2.0, 0.9, "Preprocess"),
        (5.6, 4.2, 2.2, 0.9, "TensorRT\nFP16 / INT8"),
        (8.4, 4.2, 2.5, 0.9, "Detection +\nGeometry"),
        (5.6, 2.2, 2.2, 0.9, "Pose +\nUncertainty"),
        (8.4, 2.2, 2.5, 0.9, "SOP Checks"),
        (5.6, 0.5, 2.2, 0.9, "Temporal\nStability"),
        (8.4, 0.5, 2.5, 0.9, "Assessment JSON"),
    ]
    for x, y, w, h, label in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03",
            linewidth=1.2
        )
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=10)

    arrows = [
        ((2.4, 4.65), (3.0, 4.65)),
        ((5.0, 4.65), (5.6, 4.65)),
        ((7.8, 4.65), (8.4, 4.65)),
        ((9.65, 4.2), (6.7, 3.1)),
        ((7.8, 2.65), (8.4, 2.65)),
        ((9.65, 2.2), (6.7, 1.4)),
        ((7.8, 0.95), (8.4, 0.95)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=15,
            linewidth=1.1
        ))

    ax.text(
        0.4, 5.55,
        "Section 4 deployment architecture — target Jetson path",
        fontsize=14, fontweight="bold", ha="left"
    )
    ax.text(
        0.4, 0.05,
        "Jetson performance is an acceptance test, not a measured result in this submission.",
        fontsize=9, ha="left"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def latency(path: Path, out: Path) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text())
    values = data.get("wall_clock_ms", data.get("latency_ms", {}))
    labels = ["p50", "p95", "p99"]
    ys = [values.get(k) for k in labels]
    if any(v is None for v in ys):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, ys)
    ax.axhline(1000.0 / 15.0, linestyle="--", linewidth=1.2)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Measured inference latency on available hardware")
    ax.text(
        0.98, 0.96, "Dashed line = 15 FPS budget (66.7 ms)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-json", default="section4_deployment/artifacts/latency_results.json")
    ap.add_argument("--output-dir", default="section4_deployment/artifacts/figures")
    args = ap.parse_args()
    out = Path(args.output_dir)
    architecture(out / "section4_deployment_architecture.png")
    latency(Path(args.latency_json), out / "measured_latency.png")
    print(f"Figures written under {out}")


if __name__ == "__main__":
    main()
