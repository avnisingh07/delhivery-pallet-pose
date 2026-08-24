# Section 4 — Deployment & Robustness

This directory contains the deployment evidence and robustness design for the Delhivery pallet perception assignment.

## Key measured results

All measurements are from an Apple MacBook Air M2. Jetson Orin Nano was unavailable.

| Benchmark | P50 | P95 | Mean FPS |
|---|---:|---:|---:|
| Section 3 end-to-end, M2 CPU | 438.73 ms | 515.38 ms | 2.21 |
| Geometry PyTorch, M2 CPU | 184.81 ms | 298.44 ms | 4.90 |
| Geometry PyTorch, M2 MPS | 66.73 ms | 136.01 ms | 12.75 |
| Geometry ONNX FP32, M2 ORT CPU | 267.28 ms | 473.48 ms | 3.23 |
| Geometry ONNX INT8, M2 ORT CPU | 181.78 ms | 264.81 ms | 4.93 |

Target: 15 FPS = 66.67 ms/frame.

## Quantisation

- FP32 ONNX: 38.7 MB
- INT8 ONNX: 10.3 MB
- Size reduction: ~73.4%
- Mask mAP50-95: 0.8747 → 0.8586
- INT8 calibration: 15 validation images

The INT8 result is a constrained experiment, not a production calibration claim.

## Files

- `SECTION4_REPORT.md` — final report section
- `QUANTIZATION_RESULTS.md` — FP32/INT8 accuracy and latency
- `JETSON_DEPLOYMENT_PLAN.md` — target-device plan and unvalidated items
- `TEMPORAL_DESIGN.md` — temporal aggregation design
- `artifacts/section4_measured_results.json` — reproducible measured results
- `artifacts/failure_state_contract.json` — downstream failure contract
- `artifacts/figures/` — report figures

See the repository-level README for the complete assignment.
