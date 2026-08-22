from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "evaluation"
PLOTS = EVAL / "plots"

PLOTS.mkdir(
    parents=True,
    exist_ok=True,
)

df = pd.read_csv(
    EVAL / "pose_evaluation.csv"
)

# ---------------------------------------------------------
# Landmark localization sensitivity
# ---------------------------------------------------------

d = df[
    df["experiment"]
    == "landmark_noise"
].copy()

if not d.empty:

    g = (
        d.groupby("sigma_px")
        .agg(
            translation_p95=(
                "translation_m",
                lambda x: x.quantile(0.95),
            ),
            rotation_p95=(
                "rotation_deg",
                lambda x: x.quantile(0.95),
            ),
            pass_rate=(
                "pass",
                "mean",
            ),
        )
        .reset_index()
    )

    plt.figure()

    plt.plot(
        g["sigma_px"],
        g["translation_p95"],
        marker="o",
    )

    plt.axhline(
        0.02,
        linestyle="--",
    )

    plt.xlabel(
        "Endpoint localization noise (px)"
    )

    plt.ylabel(
        "P95 translation error (m)"
    )

    plt.title(
        "Translation sensitivity to landmark noise"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS
        / "translation_vs_landmark_noise.png",
        dpi=160,
    )

    plt.close()

    plt.figure()

    plt.plot(
        g["sigma_px"],
        g["rotation_p95"],
        marker="o",
    )

    plt.axhline(
        3.0,
        linestyle="--",
    )

    plt.xlabel(
        "Endpoint localization noise (px)"
    )

    plt.ylabel(
        "P95 rotation error (deg)"
    )

    plt.title(
        "Rotation sensitivity to landmark noise"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS
        / "rotation_vs_landmark_noise.png",
        dpi=160,
    )

    plt.close()


# ---------------------------------------------------------
# Camera height sensitivity
# ---------------------------------------------------------

d = df[
    df["experiment"]
    == "height_sensitivity"
].copy()

if not d.empty:

    g = (
        d.groupby(
            [
                "height_error_m",
                "distance_m",
            ]
        )
        ["translation_m"]
        .quantile(0.95)
        .reset_index()
    )

    for distance in sorted(
        g["distance_m"].unique()
    ):

        x = g[
            g["distance_m"]
            == distance
        ]

        plt.figure()

        plt.plot(
            x["height_error_m"],
            x["translation_m"],
            marker="o",
        )

        plt.axhline(
            0.02,
            linestyle="--",
        )

        plt.xlabel(
            "Camera height error (m)"
        )

        plt.ylabel(
            "P95 translation error (m)"
        )

        plt.title(
            f"Height sensitivity at {distance:g} m"
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS
            / f"height_sensitivity_{distance:g}m.png",
            dpi=160,
        )

        plt.close()


# ---------------------------------------------------------
# Camera tilt sensitivity
# ---------------------------------------------------------

d = df[
    df["experiment"]
    == "tilt_sensitivity"
].copy()

if not d.empty:

    g = (
        d.groupby(
            [
                "tilt_error_deg",
                "distance_m",
            ]
        )
        ["translation_m"]
        .quantile(0.95)
        .reset_index()
    )

    for distance in sorted(
        g["distance_m"].unique()
    ):

        x = g[
            g["distance_m"]
            == distance
        ]

        plt.figure()

        plt.plot(
            x["tilt_error_deg"],
            x["translation_m"],
            marker="o",
        )

        plt.axhline(
            0.02,
            linestyle="--",
        )

        plt.xlabel(
            "Camera tilt error (deg)"
        )

        plt.ylabel(
            "P95 translation error (m)"
        )

        plt.title(
            f"Tilt sensitivity at {distance:g} m"
        )

        plt.tight_layout()

        plt.savefig(
            PLOTS
            / f"tilt_sensitivity_translation_{distance:g}m.png",
            dpi=160,
        )

        plt.close()


# ---------------------------------------------------------
# Operating envelope
# ---------------------------------------------------------

d = df[
    df["experiment"]
    == "operating_envelope"
].copy()

if not d.empty:

    pivot = d.pivot(
        index="distance_m",
        columns="view_angle_deg",
        values="pass_rate",
    )

    plt.figure()

    plt.imshow(
        pivot.values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns,
    )

    plt.yticks(
        range(len(pivot.index)),
        pivot.index,
    )

    plt.xlabel(
        "Pallet yaw / viewing angle (deg)"
    )

    plt.ylabel(
        "Range (m)"
    )

    plt.colorbar(
        label="Pass rate"
    )

    plt.title(
        "Pose estimation operating envelope"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS
        / "operating_envelope_heatmap.png",
        dpi=160,
    )

    plt.close()


print(
    f"plots: {PLOTS}"
)
