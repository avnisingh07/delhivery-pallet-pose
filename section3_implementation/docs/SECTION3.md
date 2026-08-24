# Section 3 — Load Analysis & SOP Checks

## Scope

This module consumes the Section 2 pallet pose and performs conservative SOP-PAL-03
load analysis from a single RGB side view.

The system does **not** assume all eight rules are observable.

### Rule triage

| Rule | Status | Implementation |
|---|---|---|
| 1. Overhang <= 3 cm | Partial | Visible box/pallet geometry |
| 2. Height <= 1.8 m | Partial | Manual until Section 2 vertical metric interface is wired |
| 3. Box rotation <= 15 deg | Partial | Visible box geometry |
| 4. Larger below smaller | Manual | Hidden layers cannot be verified |
| 5. Stretch wrapped | Manual | Single RGB side view cannot prove complete wrapping |
| 6. No visible damage | Partial | Conservative visual/manual treatment; no damage CNN |
| 7. Centroid <= 10 cm | Partial | Visible geometric centroid proxy, not mass centroid |
| 8. Pallet undamaged | Partial | Visible surfaces only |

## Model decision

Use one pretrained YOLOE-Seg model for box instances. No Section-3 training is
performed initially because the available project datasets are pallet/geometry
datasets, not a validated box/SOP/damage dataset.

YOLOE is configured with text prompts:
- cardboard box
- carton
- package
- box

The model provides box detections and instance masks.

## Confidence

The confidence field is an **evidence-quality score**, not a calibrated probability.

For geometry-dependent checks:

C = C_detection * C_geometry * C_observability * C_pose^alpha

The pose exponent is rule-specific because some rules depend strongly on pallet pose
(overhang, rotation, centroid) while others do not (wrapping).

## Threshold policy

For a scalar measurement x with uncertainty sigma:

PASS if x + 2 sigma <= threshold
FAIL if x - 2 sigma > threshold
otherwise MANUAL_INSPECTION

For partial-observability rules, absence of an observed violation does not automatically
become PASS.

## Overall verdict

FAIL:
  at least one high-confidence observable violation.

MANUAL_INSPECTION:
  no high-confidence violation, but at least one required rule remains unverified.

PASS:
  only when all required rules are sufficiently verified.

With a one-sided camera, MANUAL_INSPECTION is therefore expected for many pallets.

## Section 2 interface

Expected pose JSON contains:

{
  "pose": {
    "x_m": ...,
    "y_m": ...,
    "theta_deg": ...,
    "position_uncertainty_m": ...,
    "orientation_uncertainty_deg": ...
  },
  "pose_quality": {
    "status": "...",
    "geometry_confidence": ...
  },
  "pallet": {
    "length_m": 1.20,
    "width_m": 1.00,
    "top_z_m": ...
  },
  "camera": {
    "K": [[...], [...], [...]],
    "T_world_camera": [[...], [...], [...], [...]],
    "image_width": ...,
    "image_height": ...
  }
}

`T_world_camera` maps camera-frame points into the Section-2 floor/world frame.

## Important limitation

The current scaffold intentionally leaves Rule 2 (metric height) conservative until the
actual Section-2 vertical camera transform is connected. Do not infer metric height from
pixel height.
