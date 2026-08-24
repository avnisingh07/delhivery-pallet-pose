
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

FAILURE_STATES = (
    "OK",
    "NO_DETECTION",
    "LOW_DETECTION_CONFIDENCE",
    "GEOMETRY_UNRELIABLE",
    "POSE_UNRELIABLE",
    "TEMPORALLY_UNSTABLE",
    "SOP_UNCERTAIN",
    "INPUT_INVALID",
    "INFERENCE_ERROR",
)

@dataclass(frozen=True)
class FailureState:
    status: str
    reason: str
    retryable: bool
    safe_action: str

def make_failure(status: str, reason: str, *, retryable: bool = True,
                 safe_action: str = "MANUAL_INSPECTION") -> dict[str, Any]:
    if status not in FAILURE_STATES:
        raise ValueError(f"Unknown failure status: {status}")
    return {
        "status": status,
        "reason": reason,
        "retryable": bool(retryable),
        "safe_action": safe_action,
    }

def validate_assessment_contract(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    status = payload.get("status")
    if status not in FAILURE_STATES:
        errors.append("status must be one of the documented failure states")
    if status == "OK":
        pose = payload.get("pose")
        if not isinstance(pose, dict):
            errors.append("OK assessment requires a pose object")
        else:
            for key in ("x_m", "y_m", "theta_deg"):
                if pose.get(key) is None:
                    errors.append(f"OK assessment requires pose.{key}")
        if "overall_verdict" not in payload:
            errors.append("OK assessment requires overall_verdict")
    else:
        if not payload.get("failure"):
            errors.append("failed assessment requires a failure object")
    return errors
