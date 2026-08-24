#!/usr/bin/env python3
"""Compare FP32 and INT8 segmentation models on the same held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def extract_metrics(metrics):
    out = {}

    if hasattr(metrics, "results_dict"):
        for key, value in metrics.results_dict.items():
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                pass

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--int8", required=True)
    ap.add_argument("--data-yaml", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--output-json",
        default="section4_deployment/artifacts/quantization_accuracy.json",
    )
    args = ap.parse_args()

    # Explicitly specify segmentation because the geometry model is YOLO11-seg.
    fp32_model = YOLO(args.fp32, task="segment")
    int8_model = YOLO(args.int8, task="segment")

    fp32_metrics = fp32_model.val(
        data=args.data_yaml,
        split=args.split,
        verbose=False,
    )

    int8_metrics = int8_model.val(
        data=args.data_yaml,
        split=args.split,
        verbose=False,
    )

    fp32 = extract_metrics(fp32_metrics)
    int8 = extract_metrics(int8_metrics)

    keys = sorted(set(fp32) & set(int8))
    delta = {
        key: int8[key] - fp32[key]
        for key in keys
    }

    result = {
        "evaluation": {
            "task": "segment",
            "dataset": args.data_yaml,
            "split": args.split,
            "same_split_for_both": True,
            "calibration_note": (
                "INT8 calibration used the available validation images. "
                "The dataset contained only 15 validation images, below "
                "the >300-image recommendation reported by Ultralytics."
            ),
        },
        "fp32": fp32,
        "int8": int8,
        "delta_int8_minus_fp32": delta,
        "interpretation": (
            "Segmentation/localisation metrics are the primary comparison "
            "because geometry errors propagate into pose estimation."
        ),
        "jetson_validation": {
            "status": "NOT_MEASURED",
            "reason": (
                "This is an M2/ONNX accuracy experiment. "
                "No Jetson Orin Nano hardware was available."
            ),
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()