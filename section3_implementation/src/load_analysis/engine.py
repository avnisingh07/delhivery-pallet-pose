from __future__ import annotations

from .types import Assessment, Pose, PalletGeometry, Camera
from ..sop.rules import (
    rule_overhang,
    rule_height,
    rule_rotation,
    rule_ordering,
    rule_wrapping,
    rule_damage,
    rule_centroid,
    rule_pallet_damage,
)


def assess(
    pallet_id: str,
    detections: list[dict],
    pose: Pose,
    pallet: PalletGeometry,
    camera: Camera | None,
    image_height: int,
) -> Assessment:

    # Evaluate all eight SOP rules.
    #
    # Rules that cannot be established from the available single-view
    # geometry intentionally return MANUAL_INSPECTION rather than
    # producing unsupported PASS/FAIL decisions.

    checks = [
        rule_overhang(
            detections,
            pose,
            pallet,
            camera,
        ),

        rule_height(
            detections,
            pose,
            pallet,
            image_height,
        ),

        rule_rotation(
            detections,
            pose,
            pallet,
            camera,
        ),

        rule_ordering(
            detections,
        ),

        rule_wrapping(),

        rule_damage(
            detections,
        ),

        rule_centroid(
            detections,
            pose,
            pallet,
            camera,
        ),

        rule_pallet_damage(),
    ]

    # Conservative overall decision:
    # - Any FAIL -> FAIL
    # - Otherwise, if any check is MANUAL_INSPECTION -> MANUAL_INSPECTION
    # - PASS only if every rule is explicitly PASS.
    #
    # In this project, several rules are intentionally conservative and
    # therefore normally resolve to MANUAL_INSPECTION for a single RGB view.

    if any(r.verdict == "FAIL" for r in checks):
        overall_verdict = "FAIL"
    elif any(r.verdict == "MANUAL_INSPECTION" for r in checks):
        overall_verdict = "MANUAL_INSPECTION"
    else:
        overall_verdict = "PASS"

    overall_reason = []

    if pose.status == "UNRELIABLE":
        overall_reason.append(
            "Section 2 pose is UNRELIABLE; metric compliance cannot be "
            "established reliably."
        )

    for result in checks:
        if result.verdict != "PASS":
            overall_reason.append(
                f"Rule {result.rule_id} ({result.name}): {result.reason}"
            )

    if not overall_reason:
        overall_reason.append(
            "All implemented SOP checks passed."
        )

    return Assessment(
        pallet_id=pallet_id,

        pose={
            "x_m": pose.x_m,
            "y_m": pose.y_m,
            "theta_deg": pose.theta_deg,
            "position_uncertainty_m": pose.position_uncertainty_m,
            "orientation_uncertainty_deg": pose.orientation_uncertainty_deg,
        },

        pose_quality={
            "status": pose.status,
            "geometry_confidence": pose.quality,
        },

        sop_checks=[
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "verdict": r.verdict,
                "confidence": r.confidence,
                "observability": r.observability,
                "evidence": r.evidence,
                "reason": r.reason,
            }
            for r in checks
        ],

        overall_verdict=overall_verdict,
        overall_reason=overall_reason,

        detections=detections,
    )