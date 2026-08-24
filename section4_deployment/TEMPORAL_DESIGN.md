# Temporal Robustness Design

## Goal

Exploit the fact that a stationary pallet should occupy a consistent location across consecutive frames without adding a full multi-object tracking stack.

## Minimal implementation

Maintain a short history of the same pallet observation (recommended window: 5–10 frames) and aggregate only observations that pass the existing perception reliability gates.

For each observation:
- image-space pallet/front-mask geometry
- pose position `(x, y)` when reliable
- orientation `theta`
- confidence / reliability status

Aggregate:
- median position for robust translation;
- circular mean/median for orientation;
- median confidence;
- positional spread and angular spread as consistency measures.

## Decision logic

1. Require a minimum number of valid observations.
2. Reject observations marked `UNRELIABLE` or `SYSTEM_ERROR`.
3. Compute robust central estimates.
4. Compute temporal consistency.
5. Promote the estimate only if spatial and angular spread remain below configured tolerances.
6. Otherwise retain `MANUAL_INSPECTION` / `UNRELIABLE`.

This reduces single-frame jitter and transient false negatives while avoiding a large tracker implementation.

## What is not claimed

No temporal accuracy improvement is claimed from this design because a real video sequence with frame-level pose ground truth was not available for a controlled temporal experiment.
