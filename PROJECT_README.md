# Delhivery Pallet Pose Estimation & Load Compliance

An end-to-end computer-vision prototype for pallet perception, metric pose estimation, conservative load/SOP assessment, and deployment planning. The repository was prepared for the Delhivery AI/ML Engineer (Computer Vision) take-home assignment and is deliberately explicit about what is measured, assumed, unavailable, and unsafe to consume downstream.

> **Current real-image result:** Section 1 produces pallet-front geometry, but the saved Section 2 real-image pose fails its reprojection-quality gate (`28.40 px` versus `3.00 px`). It is therefore `UNRELIABLE`, and the Section 3 overall outcome is correctly `MANUAL_INSPECTION`—not PASS and not FAIL.

## Contents

- [System overview](#system-overview)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Section 1 — detection and front geometry](#section-1--detection-and-front-geometry)
- [Section 2 — pose estimation](#section-2--pose-estimation)
- [Section 3 — load analysis and SOP assessment](#section-3--load-analysis-and-sop-assessment)
- [Section 4 — deployment and robustness](#section-4--deployment-and-robustness)
- [Results and artifacts](#results-and-artifacts)
- [Limitations and safety behaviour](#limitations-and-safety-behaviour)
- [Reproducibility](#reproducibility)

## System overview

```text
RGB pallet image
      │
      ├── Section 1: pallet detector ────────────────► pallet bounding boxes
      │
      └── Section 1: pallet_front segmentation ─────► mask → G0/G1 front edge
                                                              │
                                                              ▼
                                              Section 2: ray/floor geometry
                                                              │
                                            (x, y, yaw, quality status)
                                                              │
                                                              ▼
                                      Section 3: YOLOE-Seg load evidence + SOP rules
                                                              │
                                                              ▼
                                  Section 4: failure contract + temporal stability gate
```

The key design principle is conservative decision-making. A missing detection, incomplete view, low-quality geometry, or unreliable pose is propagated as an explicit state rather than converted into a plausible numeric result or a positive compliance claim.

## Repository layout

| Path | Purpose |
|---|---|
| `data/` | Raw exports, processed single-class YOLO datasets, and split manifests. |
| `section1_detection/` | Dataset processing, training notebook, trained detector/segmenter, geometry extraction, metrics, and figures. |
| `section2_pose/` | Floor-plane pose estimator, uncertainty/assessment logic, synthetic evaluation, saved pose artifact, and figures. |
| `section3_implementation/` | YOLOE-Seg load detector adapter, conservative SOP rules, saved assessment, and figures. |
| `section4_deployment/` | Benchmarking, ONNX/INT8 export path, failure contract, temporal aggregation, and deployment documentation. |
| `docs/`, `runs/` | Supplementary artifacts and Ultralytics validation outputs. |

## Quick start

### Prerequisites

- Python 3.10+ recommended
- A working PyTorch/Ultralytics installation for model inference
- The packaged Section 1 model weights in `section1_detection/models/`
- The packaged YOLOE checkpoint `yoloe-11s-seg.pt` for the saved Section 3 workflow

Create an isolated environment and install the common dependencies:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r section3_implementation/requirements-section3.txt
pip install -r section4_deployment/requirements-section4.txt
```

Run commands from the repository root. Some Section 2/3/4 modules rely on package imports, so retain the indicated `PYTHONPATH` when using those commands.

### Regenerate the report figures

The figure scripts use existing artifacts and local model weights. They do not retrain models or alter the estimators/rules.

```bash
python section1_detection/scripts/generate_section1_figures.py
PYTHONPATH=. python section2_pose/scripts/generate_section2_figures.py
python section3_implementation/scripts/generate_section3_figures.py
```

Generated files are written to:

- `section1_detection/outputs/figures/`
- `section2_pose/outputs/figures/`
- `section3_implementation/outputs/figures/`

## Section 1 — detection and front geometry

Section 1 supplies the perception interface used by the rest of the project:

1. a single-class pallet detector; and
2. a single-class `pallet_front` segmentation model that yields the visible lower/front edge.

### Data

The original exports are documented in [DATASET.md](DATASET.md). Both datasets were converted to single-class YOLO datasets and split deterministically with seed `42` using source-group-aware assignment. The supplied Roboflow split was not used as the final scientific split.

| Dataset | Retained target | Images | Retained annotations |
|---|---|---:|---:|
| Dataset 1 — pallet detection | `pallet` | 920 | 10,329 |
| Dataset 2 — pallet geometry | `pallet_front` | 412 | 278 |

| Dataset | Train | Validation | Held-out test |
|---|---:|---:|---:|
| Detection images / instances | 644 / 6,929 | 141 / 1,527 | 135 / 1,873 |
| Geometry images / instances | 288 / 199 | 61 / 42 | 63 / 37 |

Dataset 1 has 378 derived source groups (264 train, 56 validation, 58 test). Dataset 2 has 412 derived filename groups (288, 61, 63); this reduces augmentation leakage but does **not** prove complete independence of the original photographic scenes.

### Models and held-out metrics

The trained weights are stored in:

- `section1_detection/models/pallet_detector_best.pt`
- `section1_detection/models/pallet_geometry_best.pt`

The following values are read from the saved training notebook's held-out evaluation:

| Model / metric | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Pallet detector (box) | 0.947 | 0.958 | 0.968 | 0.694 |
| Pallet-front model (box) | 0.947 | 1.000 | 0.981 | 0.898 |
| Pallet-front model (mask) | 0.947 | 1.000 | 0.981 | 0.887 |

### Geometry interface

[`extract_front_edge.py`](section1_detection/scripts/extract_front_edge.py) is the fixed Section 1 geometry definition:

- form a binary `pallet_front` mask;
- take the bottom 10% of mask pixels;
- `G0` is the image-left extreme and `G1` the image-right extreme;
- pass the pixel endpoints to Section 2.

This intentionally simple interface is auditable and avoids introducing a second, competing edge definition.

### Dataset utilities and training

Dataset preparation and validation scripts are in `section1_detection/scripts/`. The training workflow and saved evaluation outputs are in `section1_detection/notebooks/01_train_section1.ipynb`.

The notebook trains YOLO11-small models at 640 px:

- detection: `yolo11s.pt`, 50 epochs, patience 10;
- segmentation: `yolo11s-seg.pt`, 75 epochs, patience 15.

Training uses seed `42`. Paths in `section1_detection/configs/paths.yaml` are local-machine settings; update them before rebuilding source exports.

## Section 2 — pose estimation

Section 2 turns fixed `G0/G1` image geometry into a floor-frame candidate pose `(x, y, theta)`.

### Method

The method is implemented in [`section2_pose/src/pose_geometry.py`](section2_pose/src/pose_geometry.py):

1. cast calibrated camera rays through `G0` and `G1`;
2. intersect rays with the floor plane (`Z = 0`);
3. use the front-edge midpoint and known pallet width to recover the local edge;
4. recover pallet +X as the right-handed perpendicular to the edge;
5. translate by half the nominal pallet length to obtain the pallet centre;
6. refine `(x, y, theta)` with constrained reprojection; and
7. gate the result on reprojection quality.

The floor frame is: +X pallet front, +Y pallet left, +Z up, with origin at the camera-floor projection. `theta` is pallet +X yaw relative to floor +X.

### Explicit assumptions

The assignment did not provide physical pallet dimensions or deployment calibration. The implementation therefore uses **nominal assumptions**, not measured facts:

| Parameter | Value |
|---|---:|
| Pallet length | 1.20 m |
| Pallet width | 1.00 m |
| Deck height | 0.144 m |
| Reference image resolution | 640 × 480 px |
| Reference intrinsics | fx = fy = 450 px; cx = 320 px; cy = 240 px |
| Reference camera height / tilt | 1.20 m / 20° |
| Pose reprojection gate | 3.00 px RMS |

### Evaluation and real-image integration

The controlled pose evaluation is synthetic because physical pose ground truth and a target camera were not available. It validates the estimator procedure under known simulated poses; it is not a physical deployment validation.

| Result | Value |
|---|---:|
| Synthetic calibration RMS / P95 / max | 0.325 / 0.647 / 1.050 px |
| 1 px endpoint-noise P95 translation | 0.070 m |
| 1 px endpoint-noise P95 rotation | 4.452° |
| 1 px endpoint-noise pass rate | 43% |
| Passing synthetic envelope | 1.0 m, viewing angles 0–50° |
| Tested 2 m+ envelope | No point met the 95% acceptance criterion |

The saved real-image integration in `section2_pose/artifacts/real_pose/pose_assessment.json` contains:

```text
G0 = (395, 279) px
G1 = (497, 279) px
reprojection RMS = 28.40 px
status = UNRELIABLE
reason = REPROJECTION_RMS_EXCEEDS_3PX
```

The numeric candidate pose is deliberately **not** a valid metric result and must not be consumed as such downstream.

### Run the supplied evaluations

```bash
PYTHONPATH=. python section2_pose/evaluation/calibration_synthetic.py
PYTHONPATH=. python section2_pose/evaluation/synthetic_pose_eval.py --n 100 --envelope-trials 100
PYTHONPATH=. python section2_pose/evaluation/uncertainty_mc.py
PYTHONPATH=. python section2_pose/evaluation/plot_results.py
```

For an image with the expected 640×480 calibration convention:

```bash
PYTHONPATH=. python section2_pose/inference/run_pose.py \
  --image path/to/image.jpg \
  --conf 0.25
```

Further rationale, alternatives, sensitivities, and acceptance criteria are in [SECTION2_METHOD.md](section2_pose/SECTION2_METHOD.md).

## Section 3 — load analysis and SOP assessment

Section 3 consumes the Section 2 pose interface and applies conservative SOP-PAL-03 evidence rules to a single RGB view.

### Detection approach

[`BoxDetector`](section3_implementation/src/load_analysis/box_detector.py) uses a pretrained YOLOE-Seg model with the prompts:

```text
cardboard box, carton, package, box
```

It can return box instances and masks when the model detects them. No Section 3 detector was trained because the supplied datasets are pallet/geometry datasets, not a validated box, damage, or SOP dataset.

### SOP rules

| Rule | Threshold / scope | Observability policy |
|---|---|---|
| 1. Overhang | ≤ 3 cm | Visible box/pallet geometry only |
| 2. Height | ≤ 1.80 m | Manual until vertical metric geometry is available |
| 3. Box rotation | ≤ 15° | Visible box geometry only |
| 4. Size ordering | Larger below smaller | Manual: hidden layers cannot be established |
| 5. Stretch wrap | Complete wrapping | Manual: one RGB view cannot prove it |
| 6. Damage | No visible damage | Partial/manual: no dedicated damage model |
| 7. Centroid | ≤ 10 cm | Visible geometric proxy only, not mass centroid |
| 8. Pallet damage | No damage | Visible pallet surfaces only |

For a scalar measure `x` and uncertainty `sigma`, the decision rule is:

```text
PASS   if x + 2σ <= threshold
FAIL   if x - 2σ > threshold
MANUAL_INSPECTION otherwise
```

Absence of a visible violation is not converted into PASS when evidence is partial. Rule confidence is an evidence-quality value, not a calibrated probability:

```text
C = C_detection × C_geometry × C_observability × C_pose^alpha
```

### Saved Section 3 result

The saved assessment is [`section3_output.json`](section3_output.json). It reports:

- zero YOLOE load/box detections;
- all rules 1–8 as `MANUAL_INSPECTION`;
- overall verdict: `MANUAL_INSPECTION`;
- primary limitation: the incoming Section 2 pose is `UNRELIABLE`.

This is intentional conservative behaviour, not a compliance failure or success claim.

Run the Section 3 entry point with an image and a Section 2 pose JSON:

```bash
cd section3_implementation
python scripts/run_section3.py \
  --image ../section3_image_sample.jpg \
  --pose-json ../section2_pose/artifacts/real_pose/pose_assessment.json \
  --output-json outputs/load/pallet_001.json \
  --output-vis outputs/load/pallet_001.jpg \
  --weights ../yoloe-11s-seg.pt
```

## Section 4 — deployment and robustness

Section 4 adds a reproducible deployment boundary without changing Sections 1–3:

- model and end-to-end latency benchmarks;
- ONNX export and an optional INT8 sensitivity experiment;
- explicit failure-state contract;
- five-frame temporal aggregation for stationary pallets; and
- a Jetson Orin Nano validation plan.

### Deployment target and measurement policy

The deployment target is a Jetson Orin Nano at 15 W and ≥15 FPS, corresponding to a **66.67 ms/frame** end-to-end budget. Jetson hardware was not available, so no Jetson latency, FPS, memory, power, or thermal result is claimed.

Measured artifacts are from an Apple MacBook Air M2. They are not proxies for Jetson TensorRT throughput.

| Measured M2 result | Mean | P95 |
|---|---:|---:|
| Section 1 geometry, PyTorch CPU | 204.11 ms | 298.44 ms |
| Section 1 geometry, PyTorch MPS | 78.44 ms | 136.01 ms |
| Section 3 end-to-end CPU | 451.88 ms | 515.38 ms |
| Geometry ONNX FP32, ORT CPU | 309.80 ms | 473.48 ms |
| Geometry ONNX INT8, ORT CPU | 202.94 ms | 264.81 ms |

The optional INT8 experiment reduced ONNX model size from 38.7 MB to 10.3 MB (73.4%) and decreased mean ORT CPU latency. Its held-out geometry-mask mAP50–95 changed from 0.8747 (FP32) to 0.8586 (INT8). Only 15 calibration images were available, so this is a constrained sensitivity experiment, not production-grade INT8 calibration evidence.

### Failure and temporal behaviour

The failure contract exposes states such as `NO_DETECTION`, `GEOMETRY_UNRELIABLE`, `POSE_UNRELIABLE`, `TEMPORALLY_UNSTABLE`, and `SOP_UNCERTAIN`; a failure is never encoded as a zero pose. The safe action for uncertainty is `MANUAL_INSPECTION`.

For a stationary pallet, the temporal module uses a five-frame window, robustly aggregates valid x/y/yaw observations, and rejects windows whose position or rotation jitter exceeds the configured tolerance. It intentionally does not introduce a tracker or Kalman filter.

See [section4_deployment/README.md](section4_deployment/README.md) for benchmark commands, ONNX/INT8 workflow, and the Jetson deployment plan.

## Results and artifacts

### Report figures

| Section | Location |
|---|---|
| Section 1 dataset, prediction, geometry, and evaluation figures | `section1_detection/outputs/figures/` |
| Section 2 input, front edge, pose frame, pipeline, and evaluation figures | `section2_pose/outputs/figures/` |
| Section 3 input, detection evidence, SOP evidence, rules, and assessment figures | `section3_implementation/outputs/figures/` |
| Section 4 deployment/quantisation figures | `section4_deployment/artifacts/figures/` |

### Key machine-readable artifacts

| Artifact | Description |
|---|---|
| `section1_detection/outputs/metrics/dataset_audit.json` | Dataset audit output. |
| `section2_pose/artifacts/calibration/synthetic_calibration.json` | Synthetic/reference calibration result. |
| `section2_pose/artifacts/evaluation/pose_evaluation_summary.json` | Synthetic pose evaluation summary. |
| `section2_pose/artifacts/real_pose/pose_assessment.json` | Real-image geometry and pose-quality assessment. |
| `section3_output.json` | Saved Section 3 assessment and rule outcomes. |
| `section4_deployment/artifacts/section4_measured_results.json` | M2 benchmark and quantisation metrics. |
| `section4_deployment/artifacts/failure_state_contract.json` | Failure-aware downstream contract. |

## Limitations and safety behaviour

This repository does **not** claim the following:

- physical camera calibration for a Delhivery deployment camera;
- physical pallet-pose ground truth or measured real-world pose accuracy;
- compliance verification from hidden faces, hidden layers, or unavailable load evidence;
- a successful pose for the saved real image;
- Jetson Orin Nano benchmark, power, thermal, memory, or TensorRT performance; or
- production-grade INT8 calibration.

Before deployment, collect target-camera calibration data, physical pallet/pose ground truth, representative loading and damage data, multi-view/temporal observations, and target-device benchmarks. Continue to preserve explicit failure states and require manual inspection whenever the perception evidence is insufficient.

## Reproducibility

- Dataset split seed: `42`.
- Section 1 training configuration: stored in [`01_train_section1.ipynb`](section1_detection/notebooks/01_train_section1.ipynb).
- Section 2 evaluation code: `section2_pose/evaluation/`.
- Section 3 SOP configuration: [`section3_implementation/configs/sop.yaml`](section3_implementation/configs/sop.yaml).
- Section 4 scripts and artifact schemas: `section4_deployment/scripts/` and `section4_deployment/artifacts/`.

AI assistance was used for code scaffolding, debugging, documentation drafting, and visualization preparation. Numerical artifacts in the repository are retained as executed outputs; unavailable physical measurements are explicitly reported as unmeasured rather than estimated.
