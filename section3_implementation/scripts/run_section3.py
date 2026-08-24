#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.load_analysis.box_detector import BoxDetector
from src.load_analysis.pose_adapter import (
    load_json,
    parse_pose,
    parse_pallet,
    parse_camera,
)
from src.load_analysis.engine import assess
from src.load_analysis.visualize import draw_assessment


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--image", required=True)

    ap.add_argument(
        "--pose-json",
        required=True,
        help="Section 2 pose JSON for this pallet/image",
    )

    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-vis", required=True)

    ap.add_argument("--weights", default="yoloe-11s-seg.pt")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None)

    args = ap.parse_args()

    # -----------------------------
    # Load Section 2 pose interface
    # -----------------------------
    pose_data = load_json(args.pose_json)

    pose = parse_pose(pose_data)
    pallet = parse_pallet(pose_data)
    camera = parse_camera(pose_data)

    # -----------------------------
    # Load Section 3 detector
    # -----------------------------
    detector = BoxDetector(
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    detections = detector.predict(args.image)

    # -----------------------------
    # Load image
    # -----------------------------
    import cv2

    img = cv2.imread(args.image)

    if img is None:
        raise FileNotFoundError(args.image)

    pallet_id = Path(args.image).stem

    # -----------------------------
    # Run SOP assessment
    # -----------------------------
    result = assess(
        pallet_id=pallet_id,
        detections=detections,
        pose=pose,
        pallet=pallet,
        camera=camera,
        image_height=img.shape[0],
    )

    # -----------------------------
    # Serialize assessment
    # -----------------------------
    out = {
        "pallet_id": result.pallet_id,
        "pose": result.pose,
        "pose_quality": result.pose_quality,
        "sop_checks": result.sop_checks,
        "overall_verdict": result.overall_verdict,
        "overall_reason": result.overall_reason,
        "detections": result.detections,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(out, indent=2)
    )

    # -----------------------------
    # Visualization
    # -----------------------------
    draw_assessment(
        args.image,
        out,
        args.output_vis,
    )

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()