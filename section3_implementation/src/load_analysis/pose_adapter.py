from __future__ import annotations

import json
import numpy as np

from .types import Pose, PalletGeometry, Camera


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _get_assessment(data: dict) -> dict:
    """
    Accept both:
      - native Section 2 schema: {"assessments": [...]}
      - Section 3 flattened schema: {"pose": {...}, ...}
    """
    assessments = data.get("assessments")
    if isinstance(assessments, list) and assessments:
        return assessments[0]

    return data


def parse_pose(data: dict) -> Pose:
    a = _get_assessment(data)

    p = a.get("pose", data.get("pose", {}))

    pose_status = str(
        a.get(
            "pose_status",
            data.get("pose_quality", {}).get("status", "UNKNOWN"),
        )
    )

    reprojection_rms = a.get(
        "reprojection_rms_px",
        data.get("pose_quality", {}).get("reprojection_rms_px"),
    )

    # Section 2 does not currently export uncertainty in the real-image
    # artifact. Do not invent it.
    position_uncertainty = float(
        p.get(
            "position_uncertainty_m",
            p.get("sigma_position_m", 0.0),
        )
    )

    orientation_uncertainty = float(
        p.get(
            "orientation_uncertainty_deg",
            p.get("sigma_theta_deg", 0.0),
        )
    )

    # A reliable pose receives full quality only when Section 2 explicitly
    # says so. Unknown/unreliable poses cannot silently become quality=1.
    if pose_status.upper() == "RELIABLE":
        quality = 1.0
    else:
        quality = 0.0

    return Pose(
        x_m=float(p["x_m"]),
        y_m=float(p["y_m"]),
        theta_deg=float(
            p.get("theta_deg", p.get("yaw_deg", 0.0))
        ),
        position_uncertainty_m=position_uncertainty,
        orientation_uncertainty_deg=orientation_uncertainty,
        quality=quality,
        status=pose_status,
    )


def parse_pallet(data: dict) -> PalletGeometry:
    a = _get_assessment(data)

    g = data.get(
        "pallet",
        data.get(
            "pallet_geometry",
            a.get("pallet", a.get("pallet_geometry", {})),
        ),
    )

    return PalletGeometry(
        length_m=float(g.get("length_m", 1.20)),
        width_m=float(g.get("width_m", 1.00)),
        top_z_m=float(
            g.get(
                "top_z_m",
                g.get("deck_height_m", g.get("height_m", 0.144)),
            )
        ),
    )


def parse_camera(data: dict) -> Camera | None:
    c = data.get("camera")

    if not c:
        return None

    K = c.get("K", c.get("intrinsics"))
    T = c.get("T_world_camera", c.get("extrinsics"))

    if K is None or T is None:
        return None

    return Camera(
        K=np.asarray(K, dtype=float),
        T_world_camera=np.asarray(T, dtype=float),
        image_width=int(c.get("image_width", 0)),
        image_height=int(c.get("image_height", 0)),
    )