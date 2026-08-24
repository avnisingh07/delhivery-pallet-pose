from __future__ import annotations
import math
import numpy as np
from ..load_analysis.confidence import evidence_confidence, interval_verdict
from ..load_analysis.geometry import (
    polygon_overhang, principal_angle_deg, world_to_pallet_xy,
    min_area_rect_angle, mask_to_pallet_plane,
)
from ..load_analysis.types import Pose, PalletGeometry, Camera, RuleResult

OVERHANG_M = 0.03
HEIGHT_M = 1.80
ROTATION_DEG = 15.0
CENTROID_M = 0.10

def rule_overhang(detections, pose: Pose, pallet: PalletGeometry, camera: Camera | None):
    if camera is None:
        return RuleResult(1, "overhang", "MANUAL_INSPECTION", 0.0, "PARTIAL",
                          reason="Camera metric geometry unavailable.")
    measurements = []
    for d in detections:
        if not d.get("mask_xy"):
            continue
        try:
            poly_w = mask_to_pallet_plane(d["mask_xy"], camera, pallet.top_z_m)
            poly_p = world_to_pallet_xy(poly_w, pose.x_m, pose.y_m, pose.theta_deg)
            ov = polygon_overhang(poly_p, pallet.length_m, pallet.width_m)
            if np.isfinite(ov):
                measurements.append(ov)
        except Exception:
            continue

    if not measurements:
        return RuleResult(1, "overhang", "MANUAL_INSPECTION", 0.20, "PARTIAL",
                          reason="No reliable visible box footprint could be recovered.")

    max_ov = max(measurements)
    # Geometry sigma is deliberately conservative; it will be replaced by measured
    # Section 2 uncertainty propagation when the final interface is wired in.
    sigma = max(0.005, pose.position_uncertainty_m)
    verdict = interval_verdict(max_ov, sigma, OVERHANG_M, True)
    observability = 0.65  # one-sided camera; hidden pallet edge remains unknown
    conf = evidence_confidence(0.80, 0.85, observability, pose.quality, 1)

    if verdict == "PASS":
        verdict = "MANUAL_INSPECTION"
        reason = "No visible overhang exceeds 3 cm, but the hidden pallet side is not observable."
    elif verdict == "FAIL":
        reason = "A visible box footprint extends beyond the pallet boundary by more than 3 cm with margin."
    else:
        reason = "Visible overhang estimate overlaps the 3 cm threshold within uncertainty."

    return RuleResult(1, "overhang", verdict, conf, "PARTIAL",
                      {"max_visible_overhang_m": max_ov, "sigma_m": sigma}, reason)

def rule_height(detections, pose: Pose, pallet: PalletGeometry, image_h: int):
    if not detections:
        return RuleResult(2, "height", "MANUAL_INSPECTION", 0.0, "NONE",
                          reason="No load detections.")
    # Height estimation requires camera vertical geometry. This rule is intentionally
    # not guessed from pixel height. The runner marks it manual until Section 2 exports
    # the vertical metric transform / calibrated ray-plane interface.
    return RuleResult(2, "height", "MANUAL_INSPECTION", 0.0, "PARTIAL",
                      reason="Section 2 vertical metric transform is required; pixel height alone is not metric.")

def rule_rotation(detections, pose: Pose, pallet: PalletGeometry, camera: Camera | None):
    if camera is None:
        return RuleResult(3, "box_rotation", "MANUAL_INSPECTION", 0.0, "PARTIAL",
                          reason="Camera metric geometry unavailable.")
    angles = []
    for d in detections:
        if not d.get("mask_xy"):
            continue
        try:
            poly_w = mask_to_pallet_plane(d["mask_xy"], camera, pallet.top_z_m)
            poly_p = world_to_pallet_xy(poly_w, pose.x_m, pose.y_m, pose.theta_deg)
            a, _, _ = min_area_rect_angle(poly_p)
            if np.isfinite(a):
                angles.append(principal_angle_deg(a, 0.0))
        except Exception:
            continue

    if not angles:
        return RuleResult(3, "box_rotation", "MANUAL_INSPECTION", 0.15, "PARTIAL",
                          reason="No reliable box orientation could be recovered.")
    worst = max(angles)
    sigma = max(2.0, pose.orientation_uncertainty_deg)
    verdict = interval_verdict(worst, sigma, ROTATION_DEG, True)
    conf = evidence_confidence(0.80, 0.75, 0.70, pose.quality, 3)
    if verdict == "PASS":
        reason = "All measured visible boxes are within 15 degrees; hidden boxes remain unverified."
        verdict = "MANUAL_INSPECTION"
    elif verdict == "FAIL":
        reason = "At least one visible box is rotated beyond 15 degrees with uncertainty margin."
    else:
        reason = "Worst visible box rotation overlaps the 15 degree threshold."
    return RuleResult(3, "box_rotation", verdict, conf, "PARTIAL",
                      {"max_visible_rotation_deg": worst, "sigma_deg": sigma}, reason)

def rule_ordering(detections):
    return RuleResult(4, "size_ordering", "MANUAL_INSPECTION", 0.0, "NONE",
                      reason="Single side view cannot establish the size ordering of hidden boxes/layers.")

def rule_wrapping():
    return RuleResult(5, "stretch_wrap", "MANUAL_INSPECTION", 0.0, "NONE",
                      reason="A single RGB side view cannot establish complete stretch wrapping.")

def rule_damage(detections):
    # No learned damage classifier is introduced. Only obvious geometric evidence
    # should be used later; absence of visible damage is not proof of no damage.
    return RuleResult(6, "damage", "MANUAL_INSPECTION", 0.20, "PARTIAL",
                      reason="No dedicated damage model; hidden surfaces and subtle crush damage are unobservable.")

def rule_centroid(detections, pose: Pose, pallet: PalletGeometry, camera: Camera | None):
    if camera is None or not detections:
        return RuleResult(7, "centroid", "MANUAL_INSPECTION", 0.0, "PARTIAL",
                          reason="Camera geometry or box detections unavailable.")
    pts = []
    for d in detections:
        if not d.get("mask_xy"):
            continue
        try:
            poly_w = mask_to_pallet_plane(d["mask_xy"], camera, pallet.top_z_m)
            if len(poly_w):
                pts.append(poly_w.mean(axis=0))
        except Exception:
            continue
    if not pts:
        return RuleResult(7, "centroid", "MANUAL_INSPECTION", 0.10, "PARTIAL",
                          reason="No visible load geometry available.")
    centroid_w = np.mean(np.asarray(pts), axis=0)
    centroid_p = world_to_pallet_xy(
        centroid_w.reshape(1, 2), pose.x_m, pose.y_m, pose.theta_deg
    )[0]
    distance = float(np.linalg.norm(centroid_p))
    sigma = max(0.03, pose.position_uncertainty_m)
    verdict = interval_verdict(distance, sigma, CENTROID_M, True)
    conf = evidence_confidence(0.75, 0.55, 0.50, pose.quality, 7)
    if verdict == "PASS":
        verdict = "MANUAL_INSPECTION"
        reason = "Visible geometric centroid is within 10 cm, but hidden mass and box weights are unknown."
    elif verdict == "FAIL":
        reason = "Visible geometric load centroid is outside the 10 cm tolerance with uncertainty margin."
    else:
        reason = "Centroid estimate overlaps the 10 cm tolerance."
    return RuleResult(7, "centroid", verdict, conf, "PARTIAL",
                      {"visible_geometric_centroid_offset_m": distance, "sigma_m": sigma}, reason)

def rule_pallet_damage():
    return RuleResult(8, "pallet_damage", "MANUAL_INSPECTION", 0.20, "PARTIAL",
                      reason="Only the visible pallet structure can be inspected from one side.")
