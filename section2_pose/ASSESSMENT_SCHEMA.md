# Section 2 Assessment Interface

Downstream consumers must only use metric pose when `pose_status == VALID`.

Example failed assessment:

```json
{
  "schema_version": "section2.v1",
  "instance_id": 0,
  "pose_status": "UNRELIABLE",
  "failure_reason": "REPROJECTION_RMS_EXCEEDS_3PX",
  "pose": null,
  "uncertainty": {
    "status": "NOT_VALID_FOR_DEPLOYMENT",
    "x_m": null,
    "y_m": null,
    "theta_deg": null,
    "source": "physical_camera_calibration_unavailable"
  },
  "quality": {
    "detection_confidence": 0.890892,
    "reprojection_rms_px": 28.3988
  },
  "face_identity": {
    "front": "VISIBLE_FRONT",
    "rear": "OPPOSITE_FACE",
    "left": "AMBIGUOUS",
    "right": "AMBIGUOUS"
  }
}
```

For `VALID`, pose is:
```json
{
  "frame": "floor",
  "x_m": 0.0,
  "y_m": 0.0,
  "theta_deg": 0.0,
  "face_identity": {
    "front": "VISIBLE_FRONT",
    "rear": "OPPOSITE_FACE",
    "left": "AMBIGUOUS",
    "right": "AMBIGUOUS"
  }
}
```

Rule:
- `VALID` -> pose may be consumed for metric compliance.
- `CONDITIONAL` -> numeric pose exists but deployment assumptions remain unresolved; downstream should not silently treat it as validated.
- `UNRELIABLE` -> pose must not be used for metric compliance.
