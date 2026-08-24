# Jetson Orin Nano Deployment Plan

## Hardware target

- Jetson Orin Nano
- Target power envelope: 15 W
- Required throughput: >=15 FPS
- Required frame budget: 66.67 ms/frame

## Validation status

Jetson hardware was not available during development. Therefore:
- no Jetson FPS is reported;
- no Jetson latency is reported;
- no Jetson GPU-memory number is reported;
- no Jetson power measurement is reported.

All quantitative measurements in Section 4 are from an Apple MacBook Air M2.

## Expected deployment path

PyTorch model
→ ONNX export
→ TensorRT engine on Jetson
→ FP16 or INT8 execution
→ application-level post-processing
→ pose reliability gate
→ SOP/load analysis
→ temporal aggregation
→ machine-readable assessment contract.

## Expected changes relative to the M2 benchmark

The M2 ONNX CPU measurements are not a proxy for Jetson TensorRT performance. The deployment target should use TensorRT because it is the Jetson-native inference backend and can exploit the device GPU and optimized kernels.

The expected engineering actions are:
1. Export the validated segmentation model to ONNX.
2. Build a TensorRT engine on the target Jetson so the engine is compiled for the actual device/software stack.
3. Prefer FP16 as the initial deployment mode.
4. Evaluate INT8 only after collecting a sufficiently representative calibration set.
5. Benchmark end-to-end latency on the Jetson, including preprocessing, inference and postprocessing.
6. Validate p50/p95 latency rather than reporting only average FPS.
7. Verify the complete 66.67 ms/frame budget at the application level.

## What remains unvalidated

The target-device throughput is an open validation item. A final deployment decision requires physical Jetson benchmarking.
