
# Section 4 — Deployment & Robustness

This directory is a drop-in Section 4 implementation for the Delhivery Pallet Pose
Estimation & Load Compliance project.

## Scope

Sections 1–3 remain unchanged.

The existing interfaces are reused:

- Section 1 geometry model: `section1_detection/models/pallet_geometry_best.pt`
- Section 1 detector: `section1_detection/models/pallet_detector_best.pt`
- Section 2 pose artifact: `section2_pose/artifacts/real_pose/pose_assessment.json`
- Section 3 entry point: `section3_implementation/scripts/run_section3.py`
- Section 3 load model: `yoloe-11s-seg.pt`

The current real-image Section 2 result is intentionally unreliable because its
reprojection RMS is 28.40 px while the pose gate is 3 px. Section 3 therefore remains
conservative and returns `MANUAL_INSPECTION`.

## Section 4 requirements addressed

1. Latency measurement on hardware actually available.
2. Separation of measured results from Jetson expectations.
3. ONNX export path.
4. Optional INT8 post-training static quantisation experiment.
5. Failure-aware downstream contract.
6. Lightweight stationary-pallet temporal aggregation.
7. Reproducible artifacts in JSON/CSV.
8. Report-ready architecture and latency figures.

## Target deployment

The assignment target is a Jetson Orin Nano at 15 W and >=15 FPS.

The corresponding frame budget is:

`1000 / 15 = 66.67 ms/frame`

No Jetson benchmark is claimed in this repository because the target hardware was not
available during development.

NVIDIA documents the Orin Nano 8 GB module as supporting a 7–15 W power range and
the current JetPack documentation lists a 15 W reference mode. The hardware
characteristics are used only to explain the deployment target, not to infer FPS.

## Measured vs expected

### Can be measured on the MacBook Air M2

- PyTorch/Ultralytics CPU latency
- PyTorch/Ultralytics MPS latency
- ONNX Runtime CPU latency
- Section 3 detector latency
- Section 3 downstream rule-engine latency
- end-to-end Section 3 latency
- process RSS
- FP32 vs INT8 model accuracy on the same held-out split, if INT8 export succeeds

### Cannot be measured honestly without Jetson

- Jetson Orin Nano FPS
- Jetson latency
- Jetson GPU memory
- Jetson power draw
- Jetson thermal throttling
- TensorRT FP16/INT8 throughput

These remain deployment acceptance tests.

## Recommended execution

From repository root:

```bash
# 1. Existing Section 3 baseline
PYTHONPATH=section3_implementation python section4_deployment/scripts/benchmark_section3.py   --image section3_image_sample.jpg   --pose-json section2_pose/artifacts/real_pose/pose_assessment.json   --weights yoloe-11s-seg.pt   --device cpu

# 2. Section 1 geometry model on M2 CPU
python section4_deployment/scripts/benchmark_models.py   --weights section1_detection/models/pallet_geometry_best.pt   --source section3_image_sample.jpg   --imgsz 640   --device cpu

# 3. Section 1 geometry model on M2 GPU via PyTorch MPS
python section4_deployment/scripts/benchmark_models.py   --weights section1_detection/models/pallet_geometry_best.pt   --source section3_image_sample.jpg   --imgsz 640   --device mps

# 4. Export the geometry model to ONNX
python section4_deployment/scripts/export_onnx.py   --weights section1_detection/models/pallet_geometry_best.pt   --imgsz 640

# 5. Benchmark ONNX Runtime on the Mac CPU
python section4_deployment/scripts/benchmark_onnx.py   --model section4_deployment/artifacts/pallet_geometry_best.onnx   --image section3_image_sample.jpg   --imgsz 640

# 6. Optional: INT8 export
python section4_deployment/scripts/export_int8.py   --weights section1_detection/models/pallet_geometry_best.pt   --data-yaml data/processed/geometry/data_colab.yaml   --imgsz 640   --fraction 0.25

# 7. Optional: quantify INT8 accuracy cost
python section4_deployment/scripts/compare_quantization.py   --fp32 section4_deployment/artifacts/pallet_geometry_best.onnx   --int8 section4_deployment/artifacts/pallet_geometry_best_int8.onnx   --data-yaml data/processed/geometry/data_colab.yaml   --split test

# 8. Check failure propagation and the downstream contract
PYTHONPATH=section4_deployment:section3_implementation python section4_deployment/scripts/robustness_checks.py   --pose-json section2_pose/artifacts/real_pose/pose_assessment.json

# 9. Wrap the existing Section 3 result in the Section 4 contract
PYTHONPATH=section4_deployment:section3_implementation python section4_deployment/scripts/emit_contract.py   --assessment-json section3_output.json   --output-json section4_deployment/artifacts/assessment_contract.json

# 10. Generate report figures after benchmarking
python section4_deployment/scripts/plot_section4.py
```

## Why the benchmark is structured this way

The learned detector is the main compute component. Pose geometry and the SOP rules
are deterministic downstream stages and should not be rewritten merely for perceived
performance gains.

Both model-level and downstream measurements are retained:

```text
image
  |
  +--> Section 1 geometry model
  |
  +--> Section 3 load model
              |
              v
       pose / SOP / contract
              |
              v
       temporal stability
              |
              v
       assessment JSON
```

The Section 3 benchmark measures the detector and the existing rule engine separately
and also reports their combined wall-clock latency.

## ONNX deployment boundary

The intended target path is:

```text
PyTorch checkpoint
      |
      v
     ONNX
      |
      +---- ONNX Runtime on development machine
      |
      +---- TensorRT FP16 / INT8 on Jetson Orin Nano
```

ONNX is therefore useful even though TensorRT cannot be benchmarked locally.

## Quantisation policy

INT8 is optional and deliberately limited to one experiment.

The accuracy comparison must use:

- the same held-out split;
- the same model input size;
- the same evaluation code;
- the same metrics.

The relevant cost is not only detection mAP. Because Section 1 geometry feeds Section 2
pose, localisation/segmentation quality is more important than a generic classification
metric.

If INT8 introduces a material geometry degradation, the FP16/FP32 deployment path
should be preferred until Jetson-specific measurements justify another choice.

## Failure contract

The Section 4 adapter exposes explicit states:

- `OK`
- `NO_DETECTION`
- `LOW_DETECTION_CONFIDENCE`
- `GEOMETRY_UNRELIABLE`
- `POSE_UNRELIABLE`
- `TEMPORALLY_UNSTABLE`
- `SOP_UNCERTAIN`
- `INPUT_INVALID`
- `INFERENCE_ERROR`

A failure is never encoded as a plausible zero pose.

The downstream action for uncertainty is `MANUAL_INSPECTION`.

## Temporal robustness

The pallet is stationary while the camera observes it. A fixed five-frame window is
therefore sufficient for a low-cost temporal gate.

The implementation:

- collects already-valid pose observations;
- uses a median for x/y;
- uses a wrapped-angle robust estimator for theta;
- computes position and rotation jitter;
- accepts a stable window only when the jitter is within the same operational
  tolerances used elsewhere.

No tracker or Kalman filter is introduced.

## Artifacts

After execution:

- `artifacts/latency_results.json`
- `artifacts/latency_results.csv`
- `artifacts/onnx_latency.json`
- `artifacts/section3_latency.json`
- `artifacts/quantization_accuracy.json`
- `artifacts/robustness_checks.json`
- `artifacts/assessment_contract.json`
- `artifacts/figures/section4_deployment_architecture.png`
- `artifacts/figures/measured_latency.png`

Only files actually generated by the benchmark should be committed.

## AI tool disclosure

AI assistance was used for Section 4 implementation planning, code generation,
interface review, and deployment documentation.

AI assistance was not used to fabricate or estimate benchmark measurements.

The Jetson Orin Nano was not available, so no Jetson performance number is reported.

## References

- Ultralytics export documentation: https://docs.ultralytics.com/modes/export
- Ultralytics ONNX integration: https://docs.ultralytics.com/integrations/onnx
- NVIDIA Jetson Orin Nano documentation: https://developer.nvidia.com/embedded/jetson-modules
