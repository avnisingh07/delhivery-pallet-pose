from pathlib import Path
import argparse
import csv
import json

import numpy as np

from src.pose_geometry import (
    CameraModel,
    PalletModel,
    Pose2D,
    pallet_front_edge_points,
    project_world,
    estimate_pose,
    pose_errors,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "evaluation"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


PALLET = PalletModel(
    length_m=1.20,
    width_m=1.00,
    deck_height_m=0.144,
)


def make_camera(
    height_m=1.20,
    tilt_deg=20.0,
):
    return CameraModel(
        fx=450.0,
        fy=450.0,
        cx=320.0,
        cy=240.0,
        height_m=height_m,
        tilt_deg=tilt_deg,
    )


def run_trial(
    rng,
    pose,
    true_camera,
    estimated_camera,
    pixel_noise_px,
):

    ground_truth_px = project_world(
        pallet_front_edge_points(
            pose,
            PALLET,
        ),
        true_camera,
    )

    observed_px = (
        ground_truth_px
        + rng.normal(
            0.0,
            pixel_noise_px,
            ground_truth_px.shape,
        )
    )

    estimated_pose, reprojection_rms, _ = (
        estimate_pose(
            observed_px,
            estimated_camera,
            PALLET,
        )
    )

    errors = pose_errors(
        estimated_pose,
        pose,
    )

    passed = (
        errors["translation_m"] <= 0.02
        and errors["rotation_deg"] <= 3.0
    )

    return {
        "translation_m": errors[
            "translation_m"
        ],
        "rotation_deg": errors[
            "rotation_deg"
        ],
        "ex_m": errors["ex_m"],
        "ey_m": errors["ey_m"],
        "reprojection_rms_px": (
            reprojection_rms
        ),
        "pass": int(passed),
    }


def summarize(rows):

    def percentile(
        key,
        q,
    ):

        values = np.asarray(
            [
                row[key]
                for row in rows
            ],
            dtype=float,
        )

        return float(
            np.percentile(
                values,
                q,
            )
        )

    return {
        "translation_m": {
            "median": percentile(
                "translation_m",
                50,
            ),
            "p95": percentile(
                "translation_m",
                95,
            ),
            "max": percentile(
                "translation_m",
                100,
            ),
        },

        "rotation_deg": {
            "median": percentile(
                "rotation_deg",
                50,
            ),
            "p95": percentile(
                "rotation_deg",
                95,
            ),
            "max": percentile(
                "rotation_deg",
                100,
            ),
        },

        "pass_rate": float(
            np.mean(
                [
                    row["pass"]
                    for row in rows
                ]
            )
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--envelope-trials",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    rng = np.random.default_rng(42)

    rows = []

    # ---------------------------------------------------------
    # 1. Zero-noise sanity test
    # ---------------------------------------------------------

    for _ in range(args.n):

        pose = Pose2D(
            x_m=float(
                rng.uniform(
                    1.0,
                    5.0,
                )
            ),
            y_m=float(
                rng.uniform(
                    -1.5,
                    1.5,
                )
            ),
            theta_deg=float(
                rng.uniform(
                    -50,
                    50,
                )
            ),
        )

        result = run_trial(
            rng,
            pose,
            make_camera(),
            make_camera(),
            0.0,
        )

        result.update(
            {
                "experiment":
                    "zero_noise",
                "sigma_px":
                    0.0,
            }
        )

        rows.append(result)

    # ---------------------------------------------------------
    # 2. Landmark noise
    # ---------------------------------------------------------

    for sigma in [
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
    ]:

        for _ in range(args.n):

            pose = Pose2D(
                x_m=float(
                    rng.uniform(
                        1.0,
                        5.0,
                    )
                ),
                y_m=float(
                    rng.uniform(
                        -1.5,
                        1.5,
                    )
                ),
                theta_deg=float(
                    rng.uniform(
                        -50,
                        50,
                    )
                ),
            )

            result = run_trial(
                rng,
                pose,
                make_camera(),
                make_camera(),
                sigma,
            )

            result.update(
                {
                    "experiment":
                        "landmark_noise",
                    "sigma_px":
                        sigma,
                }
            )

            rows.append(result)

    # ---------------------------------------------------------
    # 3. Camera height sensitivity
    # ---------------------------------------------------------

    for height_error in [
        -0.10,
        -0.05,
        -0.02,
        0.02,
        0.05,
        0.10,
    ]:

        for distance in [
            2.0,
            6.0,
        ]:

            for _ in range(
                args.envelope_trials
            ):

                pose = Pose2D(
                    x_m=distance,
                    y_m=0.0,
                    theta_deg=0.0,
                )

                result = run_trial(
                    rng,
                    pose,
                    make_camera(),
                    make_camera(
                        height_m=(
                            1.20
                            + height_error
                        )
                    ),
                    1.0,
                )

                result.update(
                    {
                        "experiment":
                            "height_sensitivity",
                        "height_error_m":
                            height_error,
                        "distance_m":
                            distance,
                    }
                )

                rows.append(result)

    # ---------------------------------------------------------
    # 4. Camera tilt sensitivity
    # ---------------------------------------------------------

    for tilt_error in [
        -5,
        -3,
        -2,
        -1,
        1,
        2,
        3,
        5,
    ]:

        for distance in [
            2.0,
            6.0,
        ]:

            for _ in range(
                args.envelope_trials
            ):

                pose = Pose2D(
                    x_m=distance,
                    y_m=0.0,
                    theta_deg=0.0,
                )

                result = run_trial(
                    rng,
                    pose,
                    make_camera(),
                    make_camera(
                        tilt_deg=(
                            20.0
                            + tilt_error
                        )
                    ),
                    1.0,
                )

                result.update(
                    {
                        "experiment":
                            "tilt_sensitivity",
                        "tilt_error_deg":
                            tilt_error,
                        "distance_m":
                            distance,
                    }
                )

                rows.append(result)

    # ---------------------------------------------------------
    # 5. Operating envelope
    # ---------------------------------------------------------

    envelope = []

    for distance in [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]:

        for view_angle in [
            0,
            10,
            20,
            30,
            40,
            50,
        ]:

            trials = []

            for _ in range(
                args.envelope_trials
            ):

                pose = Pose2D(
                    x_m=distance,
                    y_m=0.0,
                    theta_deg=view_angle,
                )

                trials.append(
                    run_trial(
                        rng,
                        pose,
                        make_camera(),
                        make_camera(),
                        1.0,
                    )
                )

            summary = summarize(
                trials
            )

            envelope.append(
                {
                    "experiment":
                        "operating_envelope",

                    "distance_m":
                        distance,

                    "view_angle_deg":
                        view_angle,

                    "translation_p95_m":
                        summary[
                            "translation_m"
                        ]["p95"],

                    "rotation_p95_deg":
                        summary[
                            "rotation_deg"
                        ]["p95"],

                    "pass_rate":
                        summary[
                            "pass_rate"
                        ],

                    "pass":
                        int(
                            summary[
                                "pass_rate"
                            ] >= 0.95
                        ),
                }
            )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    csv_path = (
        OUT
        / "pose_evaluation.csv"
    )

    all_rows = rows + envelope

    keys = sorted(
        {
            key
            for row in all_rows
            for key in row
        }
    )

    with csv_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=keys,
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {
        "evaluation_resolution_px": [
            640,
            480,
        ],

        "reference_camera": {
            "fx_px": 450.0,
            "fy_px": 450.0,
            "cx_px": 320.0,
            "cy_px": 240.0,
            "height_m": 1.20,
            "tilt_deg": 20.0,
            "status":
                "synthetic_reference_assumption",
        },

        "pallet": {
            "length_m": 1.20,
            "width_m": 1.00,
            "deck_height_m": 0.144,
        },

        "acceptance": {
            "translation_m":
                0.02,
            "rotation_deg":
                3.0,
            "operating_envelope_pass_rate":
                0.95,
        },

        "zero_noise":
            summarize(
                [
                    row
                    for row in rows
                    if row[
                        "experiment"
                    ]
                    == "zero_noise"
                ]
            ),

        "landmark_noise": {},

        "operating_envelope": {
            "rows":
                len(envelope),
        },

        "height_sensitivity": {
            "rows":
                len(
                    [
                        r
                        for r in rows
                        if r[
                            "experiment"
                        ]
                        == "height_sensitivity"
                    ]
                ),
        },

        "tilt_sensitivity": {
            "rows":
                len(
                    [
                        r
                        for r in rows
                        if r[
                            "experiment"
                        ]
                        == "tilt_sensitivity"
                    ]
                ),
        },
    }

    for sigma in [
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
    ]:

        subset = [
            row
            for row in rows
            if row[
                "experiment"
            ]
            == "landmark_noise"
            and row[
                "sigma_px"
            ]
            == sigma
        ]

        summary[
            "landmark_noise"
        ][str(sigma)] = summarize(
            subset
        )

    summary_path = (
        OUT
        / "pose_evaluation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"saved: {csv_path}"
    )


if __name__ == "__main__":
    main()
