
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
from statistics import median
from typing import Optional

@dataclass(frozen=True)
class PoseObservation:
    x_m: float
    y_m: float
    theta_deg: float
    valid: bool = True

@dataclass(frozen=True)
class TemporalResult:
    status: str
    x_m: Optional[float]
    y_m: Optional[float]
    theta_deg: Optional[float]
    position_jitter_m: Optional[float]
    orientation_jitter_deg: Optional[float]
    samples: int
    stable: bool
    reason: str

def _angle_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0

def _circular_median_deg(values: list[float]) -> float:
    if not values:
        raise ValueError("No angles")
    # Small stationary-window angles can be robustly summarized by choosing
    # the observed angle minimizing total wrapped absolute deviation.
    best = min(
        values,
        key=lambda candidate: sum(abs(_angle_diff_deg(candidate, v)) for v in values),
    )
    return float(((best + 180.0) % 360.0) - 180.0)

class StationaryPalletFilter:
    """Small fixed-window temporal gate for a stationary pallet.

    It deliberately avoids a tracker/Kalman dependency. The filter only
    aggregates already-valid pose observations and rejects unstable windows.
    """

    def __init__(self, window: int = 5,
                 max_position_jitter_m: float = 0.02,
                 max_orientation_jitter_deg: float = 3.0):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = int(window)
        self.max_position_jitter_m = float(max_position_jitter_m)
        self.max_orientation_jitter_deg = float(max_orientation_jitter_deg)
        self._history: deque[PoseObservation] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._history.clear()

    def update(self, obs: PoseObservation) -> TemporalResult:
        if obs.valid:
            self._history.append(obs)
        n = len(self._history)
        if n < self.window:
            return TemporalResult(
                "WARMUP", None, None, None, None, None, n, False,
                f"Waiting for {self.window - n} more valid observations"
            )

        xs = [o.x_m for o in self._history]
        ys = [o.y_m for o in self._history]
        ts = [o.theta_deg for o in self._history]

        x = float(median(xs))
        y = float(median(ys))
        theta = _circular_median_deg(ts)

        # Peak-to-peak spread is intentionally used as a conservative,
        # interpretable stability measure.
        pos_jitter = float(max(
            max(xs) - min(xs),
            max(ys) - min(ys),
        ))
        ref = theta
        rot_jitter = float(max(abs(_angle_diff_deg(t, ref)) for t in ts))

        stable = (
            pos_jitter <= self.max_position_jitter_m
            and rot_jitter <= self.max_orientation_jitter_deg
        )
        return TemporalResult(
            "STABLE" if stable else "TEMPORALLY_UNSTABLE",
            x, y, theta, pos_jitter, rot_jitter, n, stable,
            "Pose stable over stationary-frame window" if stable
            else "Pose variation exceeds temporal stability thresholds"
        )
