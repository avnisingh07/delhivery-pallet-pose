from __future__ import annotations
import math

POSE_EXPONENT = {
    1: 1.0,  # overhang
    2: 0.8,  # height
    3: 1.0,  # rotation
    4: 0.3,  # ordering
    5: 0.0,  # wrap
    6: 0.2,  # damage
    7: 1.0,  # centroid
    8: 0.4,  # pallet damage
}

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def evidence_confidence(det: float, geometry: float, observability: float,
                        pose_quality: float, rule_id: int) -> float:
    alpha = POSE_EXPONENT.get(rule_id, 1.0)
    return clamp01(
        clamp01(det)
        * clamp01(geometry)
        * clamp01(observability)
        * (clamp01(pose_quality) ** alpha)
    )

def interval_verdict(value: float, sigma: float, threshold: float,
                     higher_is_bad: bool = True, k: float = 2.0) -> str:
    lo = value - k * sigma
    hi = value + k * sigma
    if higher_is_bad:
        if hi <= threshold:
            return "PASS"
        if lo > threshold:
            return "FAIL"
    else:
        if lo >= threshold:
            return "PASS"
        if hi < threshold:
            return "FAIL"
    return "MANUAL_INSPECTION"
