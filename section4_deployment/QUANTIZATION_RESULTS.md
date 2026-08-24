# Quantisation Experiment

## Setup

- Model: `section1_detection/models/pallet_geometry_best.pt`
- Task: YOLO11 segmentation
- Input: 640x640
- FP32 artifact: `section4_deployment/artifacts/pallet_geometry_best.onnx`
- INT8 artifact: `section4_deployment/artifacts/pallet_geometry_best_int8.onnx`
- Accuracy split: 63-image held-out test split
- Calibration images: 15 validation images
- Accuracy/latency backend: ONNX Runtime CPUExecutionProvider on Apple M2

## Accuracy cost

| Metric | FP32 | INT8 | Delta |
|---|---:|---:|---:|
| Box mAP50 | 0.9806 | 0.9791 | -0.0015 |
| Box mAP50-95 | 0.8971 | 0.8770 | -0.0200 |
| Mask mAP50 | 0.9806 | 0.9791 | -0.0015 |
| Mask mAP50-95 | 0.8747 | 0.8586 | -0.0161 |
| Mask Precision | 0.9485 | 0.9228 | -0.0258 |
| Mask Recall | 0.9964 | 1.0000 | +0.0036 |

Mask mAP50-95 is the most relevant geometry metric because segmentation quality feeds the downstream pose estimate.

## Latency

| Metric | FP32 ONNX | INT8 ONNX |
|---|---:|---:|
| Mean latency | 309.80 ms | 202.94 ms |
| P50 latency | 267.28 ms | 181.78 ms |
| P95 latency | 473.48 ms | 264.81 ms |
| Mean FPS | 3.23 | 4.93 |

## Artifact size

- FP32 ONNX: 38.7 MB
- INT8 ONNX: 10.3 MB
- Size reduction: approximately 73.4%

## Limitation

Only 15 calibration images were available. Ultralytics recommends substantially more images for INT8 calibration (>300 was reported during export). Therefore this is a constrained sensitivity experiment, not production-grade INT8 calibration evidence.

No Jetson INT8 performance is claimed.
