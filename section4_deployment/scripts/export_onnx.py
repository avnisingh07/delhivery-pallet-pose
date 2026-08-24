
#!/usr/bin/env python3
"""Export an existing Section 1/3 Ultralytics model to ONNX."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=None)
    ap.add_argument("--output-dir", default="section4_deployment/artifacts")
    args = ap.parse_args()

    model = YOLO(args.weights)
    kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "dynamic": False,
        "simplify": True,
        "nms": False,
        "device": "cpu",
    }
    if args.opset is not None:
        kwargs["opset"] = args.opset

    exported = Path(model.export(**kwargs))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / exported.name
    if exported.resolve() != target.resolve():
        target.write_bytes(exported.read_bytes())

    print(f"ONNX export: {target}")
    print("Next: benchmark this artifact with benchmark_onnx.py.")
    print("Do not treat ONNX Runtime timing as Jetson timing.")


if __name__ == "__main__":
    main()
