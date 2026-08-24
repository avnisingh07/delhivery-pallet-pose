# Section 4 — Deployment & Robustness

## Scope

Section 4 evaluates deployment behaviour of the existing Sections 1–3 pipeline without changing their model or decision logic.

Target:
- Jetson Orin Nano at 15 W
- >=15 FPS
- 66.67 ms/frame budget

Development hardware:
- Apple MacBook Air M2
- macOS 14.1.1 arm64
- Python 3.13.2
- PyTorch 2.13.0
- ONNX Runtime 1.29.0

**Jetson hardware was not available. No Jetson performance, memory or power numbers are claimed.**

## 1. Latency analysis

### Existing Section 3 pipeline — M2 CPU

| Component | Mean | P50 | P95 |
|---|---:|---:|---:|
| Section 3 detector | 451.83 ms | 438.68 ms | 515.31 ms |
| Pose/SOP downstream | 0.051 ms | 0.050 ms | 0.076 ms |
| End-to-end | 451.88 ms | 438.73 ms | 515.38 ms |

Mean end-to-end throughput was 2.21 FPS.

The detector dominates the measured runtime. Pose/SOP processing is negligible by comparison and is not an optimization target.

### Section 1 geometry model

| Backend | Mean | P50 | P95 | Mean FPS |
|---|---:|---:|---:|---:|
| PyTorch CPU | 204.11 ms | 184.81 ms | 298.44 ms | 4.90 |
| PyTorch MPS | 78.44 ms | 66.73 ms | 136.01 ms | 12.75 |
| ONNX FP32 / ORT CPU | 309.80 ms | 267.28 ms | 473.48 ms | 3.23 |
| ONNX INT8 / ORT CPU | 202.94 ms | 181.78 ms | 264.81 ms | 4.93 |

The M2 MPS P50 is approximately equal to the 15 FPS frame budget, but P95 is not. Therefore sustained 15 FPS is not demonstrated on the development machine.

## 2. Target-hardware expectation

The M2 measurements are not a proxy for Jetson TensorRT.

For deployment, the expected path is:

`PyTorch → ONNX → TensorRT FP16/INT8 → Jetson Orin Nano`

TensorRT should be built and benchmarked on the physical Jetson because engine compilation, supported kernels, memory behaviour and runtime scheduling are target-device dependent.

The final target-device validation must include:
- preprocessing;
- model inference;
- postprocessing;
- application-level latency;
- p50/p95 latency;
- sustained FPS;
- GPU memory;
- power envelope.

None of those Jetson measurements were available for this submission.

## 3. Quantisation

Post-training static INT8 was exported and evaluated.

Accuracy was measured on the same 63-image held-out geometry test split.

Mask mAP50-95 changed from 0.8747 to 0.8586, a delta of -0.0161. Box mAP50-95 changed from 0.8971 to 0.8770.

On the M2 ONNX Runtime CPU backend, INT8 reduced:
- mean latency from 309.80 to 202.94 ms;
- P50 from 267.28 to 181.78 ms;
- P95 from 473.48 to 264.81 ms.

The INT8 ONNX artifact was approximately 73.4% smaller than the FP32 ONNX artifact.

Calibration used only 15 validation images, below the >300-image recommendation reported by Ultralytics. Therefore the result is a constrained quantisation sensitivity experiment, not production-grade calibration evidence.

## 4. Failure behaviour and downstream contract

The system never represents a failed pose as a valid zero-valued pose.

Machine-readable statuses:
- `VALID`
- `UNRELIABLE`
- `MANUAL_INSPECTION`
- `SYSTEM_ERROR`

A downstream robotics consumer must branch on `status` and `failure_reason`.

For example, the existing real-image pose has a high reprojection RMS and is already marked `UNRELIABLE`. Section 3 consequently remains conservative and returns `MANUAL_INSPECTION` when reliable metric geometry is unavailable.

Example contract:

```json
{
  "status": "MANUAL_INSPECTION",
  "failure_reason": "POSE_UNRELIABLE",
  "pose": {
    "status": "UNRELIABLE",
    "position_m": null,
    "orientation_deg": null,
    "uncertainty": null
  },
  "load_compliance": {
    "status": "NOT_ASSESSABLE"
  },
  "confidence": 0.0
}
```

This prevents downstream motion/planning software from interpreting missing perception as a valid pose.

## 5. Temporal robustness

A stationary pallet provides repeated observations at approximately the same world location.

A lightweight temporal layer can maintain 5–10 observations for the same pallet and aggregate only reliable observations:
- median position;
- circular aggregation of orientation;
- median confidence;
- spatial spread;
- angular spread;
- number of supporting frames.

A stable estimate is promoted only when enough observations agree within configured consistency limits. Outliers, transient detector misses and unreliable pose observations are rejected.

A full tracker is deliberately not added because it is unnecessary for the 10% deployment section and would add implementation risk.

No quantitative temporal improvement is claimed because no frame-level temporal ground truth was available.

## 6. Deployment architecture

```text
Camera
  ↓
Frame acquisition / preprocessing
  ↓
Pallet + pallet-front segmentation
  ↓
Geometry extraction
  ↓
Pose estimation
  ↓
Pose reliability gate
  ↓
Load/SOP checks
  ↓
Temporal aggregation
  ↓
Assessment contract
  ├── VALID
  └── MANUAL_INSPECTION / SYSTEM_ERROR
```

Target deployment replaces the model execution backend with a TensorRT engine on Jetson.

## 7. Engineering conclusion

The measured development-machine bottleneck is learned perception, not pose/SOP post-processing.

INT8 provides a meaningful latency reduction on the tested M2/ONNX Runtime CPU backend while causing a modest segmentation accuracy decrease. It does not meet the 15 FPS target on the measured hardware.

The correct next deployment step is physical Jetson validation, not extrapolation from the M2. The target-device result remains intentionally unclaimed because the hardware was unavailable.

## AI assistance disclosure

AI assistance was used for implementation planning, code review, debugging support, documentation drafting and deployment reasoning. All reported benchmark values were generated by executing the repository's benchmark scripts on the available Apple M2 hardware. AI-generated assumptions were checked against actual repository files, command output and measured results; in particular, the initial incorrect dataset YAML path and segmentation-task inference issue were caught and corrected.
