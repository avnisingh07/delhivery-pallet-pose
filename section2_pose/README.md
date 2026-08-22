# Section 2 — Pose Estimation

## Approach
Section 2 consumes the fixed Section 1 `G0/G1` front-edge interface. Pose is recovered by calibrated ray/floor-plane intersection followed by constrained reprojection refinement.

The nominal pallet geometry is 1.20 x 1.00 x 0.144 m and is explicitly an assumption because the assignment does not specify pallet dimensions.

## Results
Synthetic calibration at 640x480:
- RMS 0.325 px
- P95 0.647 px
- max 1.050 px

These validate the procedure only; no physical deployment calibration was available.

Synthetic operating envelope at 1 px endpoint noise:
- 1.0 m across tested 0–50 deg viewing angles
- P95 translation <= 2 cm
- P95 rotation <= 3 deg
- >=95% pass rate

No tested operating point at 2 m or beyond met the criterion.

## Real image
G0=(395,279), G1=(497,279), reprojection RMS=28.40 px.

The system returns:
`UNRELIABLE / REPROJECTION_RMS_EXCEEDS_3PX`

The candidate numeric pose is not treated as valid.

## What could not be finished
A physical camera calibration and physical pose-ground-truth campaign were impossible because no target camera, pallet, warehouse, or fiducial setup was available. Therefore the calibration and operating envelope are explicitly synthetic/reference results.

No Jetson Orin Nano benchmark is claimed.

## AI tool usage
AI assistants were used for code scaffolding, debugging, evaluation design discussion, and documentation drafting. All generated code and numerical outputs were executed and checked locally. A key issue caught during validation was the initial use of a 1280x720 intrinsic matrix against 640x480 dataset images; this was corrected before the final evaluation. No unmeasured hardware benchmark is reported.
