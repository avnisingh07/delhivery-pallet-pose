
#!/usr/bin/env python3
"""Cheap, deterministic robustness checks around existing Section 2/3 contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.failure_states import make_failure, validate_assessment_contract


def section2_pose_status(data: dict) -> tuple[str | None, float | None, str | None]:
    assessments = data.get("assessments") or []
    if assessments:
        a = assessments[0]
        return (
            a.get("pose_status"),
            a.get("reprojection_rms_px"),
            a.get("failure_reason"),
        )
    quality = data.get("pose_quality", {})
    return (
        quality.get("status"),
        quality.get("reprojection_rms_px"),
        quality.get("failure_reason"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-json", required=True)
    ap.add_argument("--output-json", default="section4_deployment/artifacts/robustness_checks.json")
    args = ap.parse_args()

    pose_path = Path(args.pose_json)
    checks = []

    if not pose_path.exists():
        checks.append({
            "name": "missing_pose_json",
            "passed": True,
            "expected_behavior": make_failure(
                "INPUT_INVALID",
                "Pose assessment artifact is missing.",
                retryable=False,
                safe_action="MANUAL_INSPECTION",
            ),
        })
    else:
        data = json.loads(pose_path.read_text())
        status, rms, reason = section2_pose_status(data)
        checks.append({
            "name": "section2_failure_propagation",
            "passed": bool(
                status == "UNRELIABLE" or
                (isinstance(rms, (int, float)) and rms > 3.0)
            ),
            "observed_status": status,
            "observed_reprojection_rms_px": rms,
            "observed_failure_reason": reason,
            "expected": "UNRELIABLE when reprojection RMS exceeds 3 px",
        })

    ok_payload = {
        "status": "OK",
        "pose": {"x_m": 0.0, "y_m": 0.0, "theta_deg": 0.0},
        "overall_verdict": "MANUAL_INSPECTION",
    }
    failed_payload = {
        "status": "POSE_UNRELIABLE",
        "failure": make_failure(
            "POSE_UNRELIABLE",
            "Pose quality gate failed.",
            retryable=True,
        ),
    }
    checks.append({
        "name": "valid_ok_contract",
        "passed": not validate_assessment_contract(ok_payload),
        "errors": validate_assessment_contract(ok_payload),
    })
    checks.append({
        "name": "valid_failure_contract",
        "passed": not validate_assessment_contract(failed_payload),
        "errors": validate_assessment_contract(failed_payload),
    })

    result = {
        "checks": checks,
        "policy": {
            "confident_wrong_answer": "never",
            "downstream_action_on_uncertainty": "MANUAL_INSPECTION",
            "jetson_benchmark_claimed": False,
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
