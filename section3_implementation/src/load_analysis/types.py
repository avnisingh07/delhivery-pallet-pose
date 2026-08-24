from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

Verdict = Literal["PASS", "FAIL", "MANUAL_INSPECTION"]

@dataclass
class Pose:
    x_m: float
    y_m: float
    theta_deg: float
    position_uncertainty_m: float = 0.0
    orientation_uncertainty_deg: float = 0.0
    quality: float = 1.0
    status: str = "GOOD"

@dataclass
class PalletGeometry:
    length_m: float = 1.20
    width_m: float = 1.00
    top_z_m: float = 0.15

@dataclass
class Camera:
    K: Any
    T_world_camera: Any
    image_width: int
    image_height: int

@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: list[float]
    mask_xy: list[list[float]] | None = None
    visible_fraction: float = 1.0

@dataclass
class RuleResult:
    rule_id: int
    name: str
    verdict: Verdict
    confidence: float
    observability: Literal["FULL", "PARTIAL", "NONE"]
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

@dataclass
class Assessment:
    pallet_id: str
    pose: dict[str, Any]
    pose_quality: dict[str, Any]
    sop_checks: list[dict[str, Any]]
    overall_verdict: Verdict
    overall_reason: list[str]
    detections: list[dict[str, Any]] = field(default_factory=list)
