
#!/usr/bin/env python3
"""Benchmark the existing Section 3 detector + rule engine without modifying it."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import psutil


TARGET_MS = 1000.0 / 15.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--pose-json", required=True)
    ap.add_argument("--weights", default="yoloe-11s-seg.pt")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--output-json", default="section4_deployment/artifacts/section3_latency.json")
    args = ap.parse_args()

    import cv2
    from src.load_analysis.box_detector import BoxDetector
    from src.load_analysis.engine import assess
    from src.load_analysis.pose_adapter import (
        load_json, parse_camera, parse_pallet, parse_pose
    )

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    pose_data = load_json(args.pose_json)
    pose = parse_pose(pose_data)
    pallet = parse_pallet(pose_data)
    camera = parse_camera(pose_data)

    detector = BoxDetector(
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    def run_one():
        t0 = time.perf_counter()
        detections = detector.predict(args.image)
        t1 = time.perf_counter()
        result = assess(
            pallet_id=Path(args.image).stem,
            detections=detections,
            pose=pose,
            pallet=pallet,
            camera=camera,
            image_height=image.shape[0],
        )
        t2 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, result

    for _ in range(args.warmup):
        run_one()

    detector_ms = []
    downstream_ms = []
    last_result = None
    for _ in range(args.iterations):
        a, b, last_result = run_one()
        detector_ms.append(a)
        downstream_ms.append(b)

    def stats(values):
        return {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }

    end_to_end = [a + b for a, b in zip(detector_ms, downstream_ms)]
    result = {
        "benchmark": "Existing Section 3 detector + rule engine",
        "measured_hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "device_argument": args.device,
        },
        "runs": {"warmup": args.warmup, "measured": args.iterations},
        "latency_ms": {
            "section3_detector": stats(detector_ms),
            "pose_sop_downstream": stats(downstream_ms),
            "end_to_end": stats(end_to_end),
        },
        "derived": {
            "end_to_end_mean_fps": float(1000.0 / np.mean(end_to_end)),
            "end_to_end_p95_fps_equivalent": float(
                1000.0 / np.percentile(end_to_end, 95)
            ),
            "target_budget_ms_at_15fps": TARGET_MS,
        },
        "last_assessment": {
            "overall_verdict": getattr(last_result, "overall_verdict", None),
            "pose_status": getattr(last_result, "pose_quality", {}).get("status")
                if last_result is not None else None,
        },
        "resource_observation": {
            "process_rss_mb": psutil.Process().memory_info().rss / (1024 ** 2),
            "note": "Process RSS only; no Jetson GPU memory/power is measured."
        },
        "jetson_validation": {
            "status": "NOT_MEASURED",
            "reason": "Jetson Orin Nano hardware was not available."
        },
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
