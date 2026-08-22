from pathlib import Path
import json
import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "calibration"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 640, 480

K_gt = np.array([
    [450.0, 0.0, 320.0],
    [0.0, 450.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

D_gt = np.zeros(5, dtype=np.float64)

board = (9, 6)
square = 0.04

obj = np.zeros(
    (board[0] * board[1], 3),
    np.float32,
)

obj[:, :2] = (
    np.mgrid[0:board[0], 0:board[1]]
    .T.reshape(-1, 2)
    * square
)

rng = np.random.default_rng(7)

objpoints = []
imgpoints = []

for _ in range(20):

    rvec = rng.normal(
        0,
        0.12,
        3,
    ).astype(np.float64)

    tvec = np.array([
        rng.uniform(-0.25, 0.25),
        rng.uniform(-0.18, 0.18),
        rng.uniform(1.2, 2.8),
    ])

    img, _ = cv2.projectPoints(
        obj,
        rvec,
        tvec,
        K_gt,
        D_gt,
    )

    noisy = (
        img.reshape(-1, 2)
        + rng.normal(
            0,
            0.25,
            img.shape[:2],
        )
    )

    objpoints.append(obj)
    imgpoints.append(
        noisy.astype(np.float32)
    )

rms, K_est, D_est, rvecs, tvecs = (
    cv2.calibrateCamera(
        objpoints,
        imgpoints,
        (W, H),
        None,
        None,
    )
)

errors = []

for obj_i, img_i, rv, tv in zip(
    objpoints,
    imgpoints,
    rvecs,
    tvecs,
):

    predicted, _ = cv2.projectPoints(
        obj_i,
        rv,
        tv,
        K_est,
        D_est,
    )

    error = np.linalg.norm(
        predicted.reshape(-1, 2) - img_i,
        axis=1,
    )

    errors.extend(
        error.tolist()
    )

result = {
    "type": "synthetic_calibration_only",

    "note": (
        "Synthetic/reference calibration at "
        "640x480. This validates the calibration "
        "procedure, not the physical target camera."
    ),

    "image_size_px": [
        W,
        H,
    ],

    "ground_truth_K": K_gt.tolist(),

    "estimated_K": K_est.tolist(),

    "ground_truth_distortion": (
        D_gt.tolist()
    ),

    "estimated_distortion": (
        D_est.ravel().tolist()
    ),

    "calibration_rms_px": float(rms),

    "per_point_reprojection_error_px": {
        "median": float(
            np.median(errors)
        ),
        "p95": float(
            np.percentile(
                errors,
                95,
            )
        ),
        "max": float(
            np.max(errors)
        ),
    },
}

path = (
    OUT
    / "synthetic_calibration.json"
)

path.write_text(
    json.dumps(
        result,
        indent=2,
    )
)

print(
    json.dumps(
        result,
        indent=2,
    )
)
