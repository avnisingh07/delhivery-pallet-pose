# Section 2 — Pose Estimation

## Objective
Convert the fixed Section 1 interface into metric floor-frame pose:
position `(x,y)` in metres, yaw `theta`, face identity, and explicit failure when a reliable pose cannot be established.

## Fixed Section 1 interface
`pallet_front segmentation -> visible lower/front-edge extraction -> G0,G1`

- G0 = left endpoint
- G1 = right endpoint

## Geometry assumptions
The assignment does not specify pallet dimensions. We explicitly assume:
- length = 1.20 m
- width = 1.00 m
- deck height = 0.144 m

These are nominal assumptions, not measured/source-provided facts. G0/G1 are modelled as floor-contact front-edge endpoints (`z=0`).

## Floor frame
- +X = pallet front
- +Y = pallet left
- +Z = up
- origin = camera floor projection
- theta = pallet +X yaw relative to floor +X

## Pose method
1. Consume G0/G1 from Section 1.
2. Convert image pixels to calibrated camera rays.
3. Transform rays into the floor frame.
4. Intersect rays with `Z=0`.
5. Compute front-edge midpoint.
6. Compute pallet +Y from G0 -> G1.
7. Recover +X as the right-handed perpendicular.
8. Translate the front midpoint by half the nominal pallet length to obtain pallet centre.
9. Refine `(x,y,theta)` using constrained image reprojection.
10. Reject the result when reprojection quality is inadequate.

This was selected because it is geometrically correct for the fixed two-endpoint interface and avoids adding an unsupported landmark detector.

## Alternatives
### PnP
Appropriate when several known 3-D landmarks have 2-D correspondences. The fixed Section 1 interface exposes only G0/G1, so PnP would require additional perception.

### Homography
Appropriate with sufficient planar correspondences. Two endpoints are insufficient; direct ray/floor intersection is simpler here.

### Keypoints / contour geometry
Could improve localization accuracy, but would require additional model/training or a new geometry extraction dependency. Given the two-day scope, this was not justified.

## Face identity
`VISIBLE_FRONT` is the observed face and `OPPOSITE_FACE` is its opposite. Left/right remain `AMBIGUOUS` because the single front-edge observation does not uniquely expose them.

## Synthetic evaluation
Because no pose ground truth exists in the supplied images, controlled synthetic trials generate known camera, pallet geometry and pose, project G0/G1, add endpoint noise, and run the same estimator. The known synthetic pose is ground truth.

## Calibration
No physical target camera was available. A synthetic 640x480 calibration experiment validates the calibration procedure only:
- RMS = 0.325 px
- median = 0.223 px
- P95 = 0.647 px
- max = 1.050 px

These are not deployment-camera calibration measurements.

## Acceptance
Target requirement:
- translation <= 0.02 m
- rotation <= 3 deg

Operating-point acceptance:
- P95 translation <= 0.02 m
- P95 rotation <= 3 deg
- trial pass rate >= 95%

## Results
At 1 px endpoint noise:
- P95 translation = 7.01 cm
- P95 rotation = 4.45 deg
- pass rate = 43%

At the tested operating envelope, the only passing points were at 1.0 m and viewing angles 0–50 deg:
- 0 deg: 1.40 cm / 1.07 deg / 100%
- 10 deg: 1.72 cm / 1.10 deg / 98%
- 20 deg: 1.56 cm / 1.00 deg / 99%
- 30 deg: 1.48 cm / 0.85 deg / 99%
- 40 deg: 1.61 cm / 1.03 deg / 100%
- 50 deg: 1.31 cm / 0.83 deg / 100%

No tested point at 2 m or beyond met the 95% criterion.

## Camera sensitivity
At 2 m, ±2 cm camera-height error produced approximately 6.2–6.6 cm P95 translation error. At 6 m it produced approximately 21.6–21.9 cm P95 translation error.

At 2 m, ±1 deg tilt error produced approximately 13–14 cm P95 translation error. At 6 m it produced approximately 70–84 cm P95 translation error. Larger tilt errors degraded performance further.

These results show that physical extrinsic calibration is essential for deployment.

## Real-image integration
For the tested 640x480 image:
- Section 1 confidence = 0.891
- G0 = (395,279)
- G1 = (497,279)
- reprojection RMS = 28.40 px

Result:
`pose_status = UNRELIABLE`
`failure_reason = REPROJECTION_RMS_EXCEEDS_3PX`

The candidate numeric pose is not a valid metric result and must not be consumed downstream.

## Uncertainty
Without a physical camera calibration and empirical endpoint covariance, a deployment-grade pose covariance cannot be claimed. The evaluation therefore quantifies endpoint-noise and camera-extrinsic sensitivity and exposes uncertainty status explicitly rather than fabricating covariance.

## Engineering conclusion
The estimator is simple, auditable, and geometrically appropriate for the fixed Section 1 interface. Its dominant limitations are endpoint localization accuracy and camera calibration. The reported 1 m envelope is synthetic/reference-only, not a physical deployment claim.
