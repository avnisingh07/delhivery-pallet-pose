
#!/usr/bin/env python3
"""Benchmark an ONNX model with ONNX Runtime on the available development machine."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import psutil


TARGET_MS = 1000.0 / 15.0


def preprocess(image_path: Path, size: int) -> np.ndarray:
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    x = image.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    return np.ascontiguousarray(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--output-json", default="section4_deployment/artifacts/onnx_latency.json")
    args = ap.parse_args()

    import onnxruntime as ort

    session = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    x = preprocess(Path(args.image), args.imgsz)

    for _ in range(args.warmup):
        session.run(None, {input_name: x})

    timings = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        timings.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(timings, 50))
    p95 = float(np.percentile(timings, 95))

    result = {
        "benchmark": "ONNX Runtime",
        "measured_hardware": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "providers": session.get_providers(),
        },
        "model": args.model,
        "input": {"imgsz": args.imgsz, "batch": 1},
        "runs": {"warmup": args.warmup, "measured": args.iterations},
        "latency_ms": {
            "mean": float(np.mean(timings)),
            "p50": p50,
            "p95": p95,
            "p99": float(np.percentile(timings, 99)),
            "min": float(np.min(timings)),
            "max": float(np.max(timings)),
        },
        "derived": {
            "mean_fps": float(1000.0 / np.mean(timings)),
            "p95_fps_equivalent": float(1000.0 / p95),
            "target_budget_ms_at_15fps": TARGET_MS,
            "p95_within_15fps_budget_on_measured_device": bool(p95 <= TARGET_MS),
        },
        "resource_observation": {
            "process_rss_mb": psutil.Process().memory_info().rss / (1024 ** 2),
            "note": "Process RSS only; no Jetson GPU memory is inferred."
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
