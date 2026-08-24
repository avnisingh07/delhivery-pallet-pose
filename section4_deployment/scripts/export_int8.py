
#!/usr/bin/env python3
"""Optional INT8 ONNX export using Ultralytics/ONNX Runtime calibration."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-yaml", required=True,
                    help="Representative dataset YAML used for calibration.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fraction", type=float, default=0.25)
    ap.add_argument("--output-dir", default="section4_deployment/artifacts")
    args = ap.parse_args()

    model = YOLO(args.weights)
    exported = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=False,
        simplify=True,
        nms=False,
        quantize=8,
        data=args.data_yaml,
        fraction=args.fraction,
        device="cpu",
    ))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / exported.name
    if exported.resolve() != target.resolve():
        target.write_bytes(exported.read_bytes())

    print(f"INT8 ONNX export: {target}")
    print("This is post-training static quantisation; validate accuracy on the same held-out split.")
    print("Do not report this as Jetson INT8 performance.")


if __name__ == "__main__":
    main()
