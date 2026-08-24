
#!/usr/bin/env python3
"""Benchmark an existing Ultralytics model without changing Sections 1–3."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import psutil
import torch
from ultralytics import YOLO


TARGET_MS = 1000.0 / 15.0


def sync(device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        fn = getattr(torch.mps, "synchronize", None)
        if fn is not None:
            fn()


def percentile(values: list[float], q: float) -> float:
    return float(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]) if q in (1, 5, 10, 50, 90, 95, 99) else float(__import__("numpy").percentile(values, q))


def device_name(device: str) -> str:
    if device == "mps":
        return "Apple M2 MPS"
    if device == "cpu":
        return "Apple M2 CPU"
    return device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="Image file or directory")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--output-json", default="section4_deployment/artifacts/latency_results.json")
    ap.add_argument("--output-csv", default="section4_deployment/artifacts/latency_results.csv")
    args = ap.parse_args()

    source = Path(args.source)
    if source.is_dir():
        image_paths = sorted(
            p for p in source.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    else:
        image_paths = [source]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {source}")

    import cv2
    images = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        images.append(image)

    model = YOLO(args.weights)
    rss_after_load_mb = psutil.Process().memory_info().rss / (1024 ** 2)

    def run_one(image_index: int):
        sync(args.device)
        t0 = time.perf_counter()
        result = model.predict(
            source=images[image_index],
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        sync(args.device)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        speed = result[0].speed if result else {}
        return wall_ms, speed

    for i in range(args.warmup):
        run_one(i % len(images))

    wall = []
    reported_pre = []
    reported_inf = []
    reported_post = []

    for i in range(args.iterations):
        wall_ms, speed = run_one(i % len(images))
        wall.append(wall_ms)
        if speed:
            reported_pre.append(float(speed.get("preprocess", 0.0)))
            reported_inf.append(float(speed.get("inference", 0.0)))
            reported_post.append(float(speed.get("postprocess", 0.0)))

    import numpy as np

    result = {
        "benchmark": "Ultralytics wall-clock inference",
        "measured_hardware": {
            "device": device_name(args.device),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "model": str(Path(args.weights)),
        "input": {"imgsz": args.imgsz, "batch": 1, "images": len(images), "source_io_excluded": True},
        "runs": {"warmup": args.warmup, "measured": args.iterations},
        "wall_clock_ms": {
            "mean": float(np.mean(wall)),
            "p50": float(np.percentile(wall, 50)),
            "p95": float(np.percentile(wall, 95)),
            "p99": float(np.percentile(wall, 99)),
            "min": float(np.min(wall)),
            "max": float(np.max(wall)),
        },
        "derived": {
            "mean_fps_on_measured_hardware": float(1000.0 / np.mean(wall)),
            "p95_fps_equivalent": float(1000.0 / np.percentile(wall, 95)),
            "target_budget_ms_at_15fps": TARGET_MS,
            "p95_within_15fps_budget_on_measured_device": bool(
                np.percentile(wall, 95) <= TARGET_MS
            ),
        },
        "ultralytics_reported_stage_ms": {
            "preprocess_p50": float(np.percentile(reported_pre, 50)) if reported_pre else None,
            "inference_p50": float(np.percentile(reported_inf, 50)) if reported_inf else None,
            "postprocess_p50": float(np.percentile(reported_post, 50)) if reported_post else None,
        },
        "resource_observation": {
            "process_rss_after_model_load_mb": rss_after_load_mb,
            "note": "RSS is process memory, not Jetson GPU memory."
        },
        "jetson_validation": {
            "status": "NOT_MEASURED",
            "reason": "Jetson Orin Nano hardware was not available."
        },
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2))

    row = {
        "model": args.weights,
        "device": device_name(args.device),
        "imgsz": args.imgsz,
        "iterations": args.iterations,
        "wall_p50_ms": result["wall_clock_ms"]["p50"],
        "wall_p95_ms": result["wall_clock_ms"]["p95"],
        "wall_p99_ms": result["wall_clock_ms"]["p99"],
        "fps_mean": result["derived"]["mean_fps_on_measured_hardware"],
        "fps_p95_equivalent": result["derived"]["p95_fps_equivalent"],
        "target_budget_ms": TARGET_MS,
        "jetson_status": "NOT_MEASURED",
    }
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
