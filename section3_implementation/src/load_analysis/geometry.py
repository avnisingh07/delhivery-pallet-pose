from __future__ import annotations
import math
import numpy as np
import cv2

def wrap_angle_deg(a: float) -> float:
    a = (a + 180.0) % 360.0 - 180.0
    return a

def principal_angle_deg(theta_deg: float, pallet_theta_deg: float) -> float:
    d = abs(wrap_angle_deg(theta_deg - pallet_theta_deg))
    return min(d, 180.0 - d)

def rotation_matrix_z(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]], dtype=float)

def world_to_pallet_xy(points_xy: np.ndarray, pose_x: float, pose_y: float,
                       pose_theta_deg: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    R = rotation_matrix_z(pose_theta_deg)
    return (pts - np.array([pose_x, pose_y])) @ R

def pallet_to_world_xy(points_xy: np.ndarray, pose_x: float, pose_y: float,
                       pose_theta_deg: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    R = rotation_matrix_z(pose_theta_deg)
    return pts @ R.T + np.array([pose_x, pose_y])

def point_to_rect_overhang(p: np.ndarray, length_m: float, width_m: float) -> float:
    x, y = float(p[0]), float(p[1])
    return max(
        0.0,
        abs(x) - length_m / 2.0,
        abs(y) - width_m / 2.0,
    )

def polygon_overhang(poly_xy: np.ndarray, length_m: float, width_m: float) -> float:
    if len(poly_xy) == 0:
        return float("nan")
    return max(point_to_rect_overhang(p, length_m, width_m) for p in poly_xy)

def contour_bottom_points(mask_xy: np.ndarray, fraction: float = 0.20) -> np.ndarray:
    """
    Keep the lowest image points of a mask. These are the best candidates for
    the visible box/pallet contact boundary, but are not assumed to expose the
    complete footprint.
    """
    pts = np.asarray(mask_xy, dtype=float)
    if len(pts) < 4:
        return pts
    y_cut = np.quantile(pts[:, 1], 1.0 - fraction)
    return pts[pts[:, 1] >= y_cut]

def min_area_rect_angle(points_xy: np.ndarray) -> tuple[float, float, float]:
    pts = np.asarray(points_xy, dtype=np.float32)
    if len(pts) < 4:
        return float("nan"), float("nan"), float("nan")
    rect = cv2.minAreaRect(pts.reshape(-1, 1, 2))
    (cx, cy), (w, h), angle = rect
    # OpenCV angle convention is tied to the selected rectangle side.
    if w < h:
        angle += 90.0
    return float(angle), float(w), float(h)

def pixel_rays_to_world_plane(
    pixels_uv: np.ndarray,
    K: np.ndarray,
    T_world_camera: np.ndarray,
    plane_z_m: float,
) -> np.ndarray:
    """
    Intersect camera rays with world Z=plane_z_m.

    T_world_camera maps camera-frame points to world-frame points.
    """
    pixels = np.asarray(pixels_uv, dtype=float)
    Kinv = np.linalg.inv(K)
    R = T_world_camera[:3, :3]
    t = T_world_camera[:3, 3]

    out = []
    for u, v in pixels:
        ray_c = Kinv @ np.array([u, v, 1.0])
        ray_c = ray_c / np.linalg.norm(ray_c)
        ray_w = R @ ray_c
        origin_w = t
        if abs(ray_w[2]) < 1e-9:
            out.append([np.nan, np.nan, np.nan])
            continue
        lam = (plane_z_m - origin_w[2]) / ray_w[2]
        p = origin_w + lam * ray_w
        out.append(p)
    return np.asarray(out)

def mask_to_pallet_plane(mask_xy: list[list[float]], camera, pallet_top_z_m: float) -> np.ndarray:
    pts_uv = np.asarray(mask_xy, dtype=float)
    bottom = contour_bottom_points(pts_uv)
    pts_w = pixel_rays_to_world_plane(
        bottom, camera.K, camera.T_world_camera, pallet_top_z_m
    )
    return pts_w[np.isfinite(pts_w).all(axis=1), :2]

def visible_fraction_from_mask(mask_xy: list[list[float]], image_w: int, image_h: int) -> float:
    """
    Heuristic only: a mask touching the image boundary is treated as potentially
    truncated. This is an observability flag, not an occlusion probability.
    """
    if not mask_xy or image_w <= 0 or image_h <= 0:
        return 0.5
    pts = np.asarray(mask_xy)
    touch = (
        (pts[:, 0] <= 1) | (pts[:, 0] >= image_w - 2) |
        (pts[:, 1] <= 1) | (pts[:, 1] >= image_h - 2)
    )
    return 0.65 if touch.any() else 1.0
