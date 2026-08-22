from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PalletModel:
    """Nominal pallet geometry used for the controlled pose evaluation.

    These dimensions are an explicit assignment assumption, not source-provided
    measurements: 1.20 m (front/back) x 1.00 m (left/right) x 0.144 m deck height.
    G0/G1 are treated as floor-contact front-edge endpoints, so z=0 for the
    Section-2 measurement interface.
    """

    length_m: float = 1.20
    width_m: float = 1.00
    deck_height_m: float = 0.144  # retained for documentation / future geometry


@dataclass(frozen=True)
class CameraModel:
    fx: float
    fy: float
    cx: float
    cy: float
    height_m: float = 1.20
    tilt_deg: float = 20.0


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    theta_deg: float


def world_to_camera_rotation(tilt_deg: float) -> np.ndarray:
    """World -> OpenCV camera coordinates.

    Floor frame:
      +X = pallet front direction / forward
      +Y = pallet left
      +Z = up

    Camera frame:
      +x = image right
      +y = image down
      +z = optical forward

    The camera is at (0, 0, height_m) and looks forward/down by tilt_deg.
    """
    a = math.radians(tilt_deg)
    return np.array(
        [
            [0.0, -1.0, 0.0],
            [-math.sin(a), 0.0, -math.cos(a)],
            [math.cos(a), 0.0, -math.sin(a)],
        ],
        dtype=np.float64,
    )


def camera_matrix(cam: CameraModel) -> np.ndarray:
    return np.array(
        [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def pallet_front_edge_points(pose: Pose2D, pallet: PalletModel) -> np.ndarray:
    """Return floor-contact front-edge endpoints [G0, G1].

    Coordinate convention:
      +X = pallet front
      +Y = pallet left

    Therefore the front edge is at local X=+L/2, and runs along -Y from
    the pallet-left endpoint to the pallet-right endpoint:
      G0 = left endpoint  = (+L/2, +W/2, 0)
      G1 = right endpoint = (+L/2, -W/2, 0)

    This matches the committed Section-1 interface: G0 is the left endpoint
    and G1 is the right endpoint of the visible lower/front edge.
    """
    th = math.radians(pose.theta_deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)

    local = np.array(
        [
            [pallet.length_m / 2.0, pallet.width_m / 2.0],
            [pallet.length_m / 2.0, -pallet.width_m / 2.0],
        ],
        dtype=np.float64,
    )
    xy = local @ R.T + np.array([pose.x_m, pose.y_m], dtype=np.float64)
    z = np.zeros((2, 1), dtype=np.float64)  # Section-1 G0/G1 are floor-contact
    return np.hstack([xy, z])


def project_world(points_w: np.ndarray, cam: CameraModel) -> np.ndarray:
    R = world_to_camera_rotation(cam.tilt_deg)
    camera_center = np.array([0.0, 0.0, cam.height_m], dtype=np.float64)
    pc = (R @ (np.asarray(points_w) - camera_center).T).T
    if np.any(pc[:, 2] <= 1e-8):
        raise ValueError("Point projects behind camera or on optical plane")
    p = (camera_matrix(cam) @ pc.T).T
    return p[:, :2] / p[:, 2:3]


def image_to_floor(points_px: np.ndarray, cam: CameraModel) -> np.ndarray:
    """Intersect image rays with the Z=0 floor plane.

    This is the primary metric conversion for Section 2 because the committed
    Section-1 G0/G1 points represent the lower/front edge on the floor plane.
    """
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    Kinv = np.linalg.inv(camera_matrix(cam))
    R = world_to_camera_rotation(cam.tilt_deg)

    # Camera-frame ray r_c = K^-1 [u,v,1]. Convert to world with R^T.
    rays_c = (Kinv @ np.column_stack([pts, np.ones(len(pts))]).T).T
    rays_w = (R.T @ rays_c.T).T
    rays_w /= np.linalg.norm(rays_w, axis=1, keepdims=True)

    C = np.array([0.0, 0.0, cam.height_m], dtype=np.float64)
    dz = rays_w[:, 2]
    if np.any(np.abs(dz) < 1e-10):
        raise ValueError("Ray is parallel to floor plane")

    lam = -C[2] / dz
    if np.any(lam <= 0):
        raise ValueError("Image ray does not intersect the floor in front of camera")

    return C[None, :] + lam[:, None] * rays_w


def estimate_pose_from_floor_edge(
    observed_px: np.ndarray, cam: CameraModel, pallet: PalletModel
) -> tuple[Pose2D, float, np.ndarray]:
    """Estimate pose analytically from the two floor-edge endpoints.

    Since G0/G1 are on Z=0 and the nominal front-edge width is known, inverse
    floor projection gives metric endpoints directly. Their midpoint is the
    front-edge midpoint. The vector G0-G1 is +Y (left direction), so the pallet
    +X/front direction is the right-handed perpendicular +X = rotate(+Y, -90°).

    Returns pose, front-edge length residual in metres, predicted image points.
    """
    observed_px = np.asarray(observed_px, dtype=np.float64).reshape(2, 2)
    if not np.all(np.isfinite(observed_px)):
        raise ValueError("Non-finite G0/G1")

    floor = image_to_floor(observed_px, cam)
    p0, p1 = floor[0, :2], floor[1, :2]
    edge = p0 - p1  # G0 -> G1 reversed; this is approximately +Y
    edge_len = float(np.linalg.norm(edge))
    if edge_len < 1e-6:
        raise ValueError("G0/G1 are coincident")

    # G0 is +Y endpoint and G1 is -Y endpoint, so G0-G1 is +Y.
    y_axis = edge / edge_len
    # For +X, +Y in a right-handed floor frame: X = -90 deg rotation of Y.
    x_axis = np.array([y_axis[1], -y_axis[0]], dtype=np.float64)

    theta = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    front_midpoint = 0.5 * (p0 + p1)

    # Front midpoint is +L/2 along pallet +X from its centre.
    center = front_midpoint - (pallet.length_m / 2.0) * x_axis
    pose = Pose2D(float(center[0]), float(center[1]), float(((theta + 180) % 360) - 180))

    predicted = project_world(pallet_front_edge_points(pose, pallet), cam)
    residual = float(np.sqrt(np.mean(np.sum((predicted - observed_px) ** 2, axis=1))))
    length_residual_m = abs(edge_len - pallet.width_m)
    return pose, length_residual_m, predicted


def reprojection_residual(
    params: np.ndarray,
    observed_px: np.ndarray,
    cam: CameraModel,
    pallet: PalletModel,
) -> np.ndarray:
    pose = Pose2D(float(params[0]), float(params[1]), float(params[2]))
    pred = project_world(pallet_front_edge_points(pose, pallet), cam)
    return (pred - observed_px).ravel()


def estimate_pose(
    observed_px: np.ndarray,
    cam: CameraModel,
    pallet: PalletModel,
    initial: Pose2D | None = None,
) -> tuple[Pose2D, float, np.ndarray]:
    """Primary Section-2 estimator.

    G0/G1 are floor-contact observations, so the estimator first performs
    calibrated ray/floor-plane intersection. A constrained reprojection
    refinement is then used only to reduce endpoint noise while preserving
    the 3-DoF physical model.
    """
    observed_px = np.asarray(observed_px, dtype=np.float64).reshape(2, 2)
    pose0, _, _ = estimate_pose_from_floor_edge(observed_px, cam, pallet)

    # Refine with known 3-D floor-edge geometry. This keeps the physical model
    # constrained to x, y, yaw and makes the image residual directly measurable.
    try:
        from scipy.optimize import least_squares

        x0 = np.array([pose0.x_m, pose0.y_m, pose0.theta_deg], dtype=np.float64)
        sol = least_squares(
            lambda p: reprojection_residual(p, observed_px, cam, pallet),
            x0,
            bounds=([0.05, -50.0, -179.9], [100.0, 50.0, 179.9]),
            loss="soft_l1",
            f_scale=2.0,
            max_nfev=200,
        )
        p = sol.x
        pose = Pose2D(float(p[0]), float(p[1]), float(((p[2] + 180) % 360) - 180))
    except Exception:
        pose = pose0

    pred = project_world(pallet_front_edge_points(pose, pallet), cam)
    rms = float(np.sqrt(np.mean(np.sum((pred - observed_px) ** 2, axis=1))))
    return pose, rms, pred


def angle_error_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def pose_errors(est: Pose2D, gt: Pose2D) -> dict:
    dx = est.x_m - gt.x_m
    dy = est.y_m - gt.y_m
    return {
        "ex_m": abs(dx),
        "ey_m": abs(dy),
        "translation_m": math.hypot(dx, dy),
        "rotation_deg": angle_error_deg(est.theta_deg, gt.theta_deg),
    }
