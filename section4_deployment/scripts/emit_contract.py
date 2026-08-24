
#!/usr/bin/env python3
"""Convert the existing Section 3 assessment into a Section 4 failure-aware envelope.

Section 1–3 outputs are not modified. This is an adapter for downstream consumers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.failure_states import make_failure, validate_assessment_contract


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assessment-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.assessment_json).read_text())
    pose_quality = data.get("pose_quality", {})
    pose_status = pose_quality.get("status")
    reproj = pose_quality.get("reprojection_rms_px")

    if pose_status == "UNRELIABLE" or (
        isinstance(reproj, (int, float)) and reproj > 3.0
    ):
        status = "POSE_UNRELIABLE"
        failure = make_failure(
            status,
            f"Section 2 pose quality gate failed"
            + (f" (reprojection RMS={reproj:.2f} px)." if isinstance(reproj, (int, float)) else "."),
            retryable=True,
            safe_action="MANUAL_INSPECTION",
        )
    elif data.get("overall_verdict") == "MANUAL_INSPECTION":
        status = "SOP_UNCERTAIN"
        failure = make_failure(
            status,
            "Existing Section 3 assessment is intentionally conservative and requires manual inspection.",
            retryable=True,
            safe_action="MANUAL_INSPECTION",
        )
    else:
        status = "OK"
        failure = None

    out = {
        "schema_version": "section4.v1",
        "status": status,
        "pallet_id": data.get("pallet_id"),
        "pose": data.get("pose"),
        "pose_quality": pose_quality,
        "sop_checks": data.get("sop_checks", []),
        "overall_verdict": data.get("overall_verdict"),
        "overall_reason": data.get("overall_reason", []),
        "failure": failure,
        "source_artifact": str(Path(args.assessment_json)),
    }

    errors = validate_assessment_contract(out)
    if errors:
        raise RuntimeError("Invalid Section 4 contract: " + "; ".join(errors))

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
