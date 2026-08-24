
from section4_deployment.src.temporal import PoseObservation, StationaryPalletFilter


def test_temporal_filter_reaches_stable_state():
    f = StationaryPalletFilter(window=5)
    result = None
    for i in range(5):
        result = f.update(PoseObservation(
            x_m=1.0 + i * 0.001,
            y_m=2.0 - i * 0.001,
            theta_deg=10.0 + i * 0.2,
        ))
    assert result is not None
    assert result.status == "STABLE"
    assert result.stable is True


def test_temporal_filter_rejects_jitter():
    f = StationaryPalletFilter(window=5, max_position_jitter_m=0.02)
    result = None
    for i in range(5):
        result = f.update(PoseObservation(
            x_m=1.0 + i * 0.01,
            y_m=2.0,
            theta_deg=10.0,
        ))
    assert result is not None
    assert result.status == "TEMPORALLY_UNSTABLE"
