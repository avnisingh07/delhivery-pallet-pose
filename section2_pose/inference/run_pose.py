from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from section2_pose.inference.section1_adapter import Section1GeometryAdapter
from section2_pose.src.pose_geometry import CameraModel, PalletModel, estimate_pose

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "section1_detection" / "models" / "pallet_geometry_best.pt"
OUT = ROOT / "section2_pose" / "artifacts" / "real_pose"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    h, w = image.shape[:2]
    if (w, h) != (640, 480):
        raise ValueError(f"Expected 640x480 image, got {w}x{h}")

    cam = CameraModel(450.0, 450.0, 320.0, 240.0, 1.20, 20.0)
    pallet = PalletModel(1.20, 1.00, 0.144)

    adapter = Section1GeometryAdapter(MODEL, args.conf)
    detections = adapter.predict(image)
    assessments = []

    for item in detections:
        base = {
            "instance_id": int(item["instance_id"]),
            "detection_confidence": float(item["confidence"]) if item["confidence"] is not None else None,
            "G0_px": list(map(int, item["G0"])) if item["G0"] is not None else None,
            "G1_px": list(map(int, item["G1"])) if item["G1"] is not None else None,
        }

        if item["G0"] is None or item["G1"] is None:
            base.update(pose_status="UNRELIABLE", failure_reason="INSUFFICIENT_GEOMETRY", pose=None)
            assessments.append(base)
            continue

        edge_px = float(np.linalg.norm(np.asarray(item["G1"], float) - np.asarray(item["G0"], float)))
        if edge_px < 20.0:
            base.update(pose_status="UNRELIABLE", failure_reason="FRONT_EDGE_TOO_SHORT", pose=None)
            assessments.append(base)
            continue

        try:
            pose, rms, predicted = estimate_pose(
                np.asarray([item["G0"], item["G1"]], dtype=float), cam, pallet
            )

            if rms > 3.0:
                status, reason = "UNRELIABLE", "REPROJECTION_RMS_EXCEEDS_3PX"
            else:
                status, reason = "CONDITIONAL", None

            base.update(
                pose_status=status,
                failure_reason=reason,
                pose={
                    "frame": "floor",
                    "x_m": float(pose.x_m),
                    "y_m": float(pose.y_m),
                    "theta_deg": float(pose.theta_deg),
                    "face_identity": {
                        "front": "VISIBLE_FRONT",
                        "rear": "OPPOSITE_FACE",
                        "left": "AMBIGUOUS",
                        "right": "AMBIGUOUS",
                    },
                },
                reprojection_rms_px=float(rms),
                predicted_G0_px=predicted[0].tolist(),
                predicted_G1_px=predicted[1].tolist(),
            )
        except Exception as exc:
            base.update(
                pose_status="UNRELIABLE",
                failure_reason=f"POSE_ESTIMATION_ERROR: {exc}",
                pose=None,
            )

        assessments.append(base)

    result = {
        "schema_version": "section2.v1",
        "evaluation_type": "real_image_integration",
        "ground_truth_available": False,
        "calibration": {
            "status": "ASSUMED_REFERENCE",
            "image_size_px": [640, 480],
            "fx_px": 450.0,
            "fy_px": 450.0,
            "cx_px": 320.0,
            "cy_px": 240.0,
            "height_m": 1.20,
            "tilt_deg": 20.0,
            "note": "Reference assumptions only; not physical deployment calibration.",
        },
        "pallet_geometry": {
            "length_m": 1.20,
            "width_m": 1.00,
            "deck_height_m": 0.144,
            "source": "explicit_nominal_assumption",
        },
        "floor_frame": {
            "origin": "camera_floor_projection",
            "x_axis": "pallet_front",
            "y_axis": "pallet_left",
            "z_axis": "up",
            "theta_definition": "pallet +X yaw relative to floor +X",
        },
        "section1_geometry_interface": {
            "source": "pallet_front_segmentation",
            "G0_definition": "image-left endpoint of lower/front edge",
            "G1_definition": "image-right endpoint of lower/front edge",
        },
        "assessments": assessments,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pose_assessment.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
