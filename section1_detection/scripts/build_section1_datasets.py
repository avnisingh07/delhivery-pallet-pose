#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import shutil
import yaml


DATASET1_NAMES = {
    0: "hole",
    1: "pallet",
}

DATASET2_NAMES = {
    0: "front",
    1: "hole",
    2: "hole_left",
    3: "hole_right",
    4: "pallet",
    5: "pallet_front",
    6: "pallet_pocket",
    7: "wood",
}


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def label_path(image_path):
    """
    Convert:
        .../train/images/foo.jpg
    to:
        .../train/labels/foo.txt
    """
    image_path = Path(image_path)
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def copy_image(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_detection(ds1_manifest, output_root):

    print("\nBuilding detection dataset...")

    counts = {
        "train": {"images": 0, "pallet_annotations": 0},
        "val": {"images": 0, "pallet_annotations": 0},
        "test": {"images": 0, "pallet_annotations": 0},
    }

    for row in ds1_manifest:

        split = row["split"]

        src_image = Path(row["image"])
        src_label = label_path(src_image)

        if not src_image.exists():
            raise FileNotFoundError(src_image)

        if not src_label.exists():
            raise FileNotFoundError(src_label)

        dst_image = (
            output_root
            / "detection"
            / "images"
            / split
            / src_image.name
        )

        dst_label = (
            output_root
            / "detection"
            / "labels"
            / split
            / f"{src_image.stem}.txt"
        )

        copy_image(src_image, dst_image)

        pallet_annotations = []

        with open(src_label) as f:

            for line in f:

                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(float(parts[0]))

                # Dataset 1:
                # 0 = hole
                # 1 = pallet
                if class_id != 1:
                    continue

                # Convert pallet -> class 0
                parts[0] = "0"

                pallet_annotations.append(
                    " ".join(parts)
                )

        dst_label.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        dst_label.write_text(
            "\n".join(pallet_annotations) +
            ("\n" if pallet_annotations else "")
        )

        counts[split]["images"] += 1
        counts[split]["pallet_annotations"] += len(
            pallet_annotations
        )

    return counts


def build_geometry(ds2_manifest, output_root):

    print("\nBuilding geometry dataset...")

    counts = {
        "train": {"images": 0, "pallet_front_annotations": 0},
        "val": {"images": 0, "pallet_front_annotations": 0},
        "test": {"images": 0, "pallet_front_annotations": 0},
    }

    for row in ds2_manifest:

        split = row["split"]

        src_image = Path(row["image"])
        src_label = label_path(src_image)

        if not src_image.exists():
            raise FileNotFoundError(src_image)

        if not src_label.exists():
            raise FileNotFoundError(src_label)

        dst_image = (
            output_root
            / "geometry"
            / "images"
            / split
            / src_image.name
        )

        dst_label = (
            output_root
            / "geometry"
            / "labels"
            / split
            / f"{src_image.stem}.txt"
        )

        copy_image(src_image, dst_image)

        pallet_front_annotations = []

        with open(src_label) as f:

            for line in f:

                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(float(parts[0]))

                # Dataset 2:
                # 5 = pallet_front
                if class_id != 5:
                    continue

                # Convert pallet_front -> class 0
                parts[0] = "0"

                pallet_front_annotations.append(
                    " ".join(parts)
                )

        dst_label.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        dst_label.write_text(
            "\n".join(pallet_front_annotations) +
            ("\n" if pallet_front_annotations else "")
        )

        counts[split]["images"] += 1
        counts[split]["pallet_front_annotations"] += len(
            pallet_front_annotations
        )

    return counts


def write_yaml(path, dataset_dir, task):

    if task == "detect":

        config = {
            "path": str(dataset_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": ["pallet"],
        }

    elif task == "segment":

        config = {
            "path": str(dataset_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,
            "names": ["pallet_front"],
        }

    with open(path, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False
        )


def print_counts(title, counts):

    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")

    for split in ["train", "val", "test"]:

        print(
            f"{split:>5}: "
            f"{counts[split]['images']} images | "
            f"{list(counts[split].values())[1]} annotations"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset1-manifest",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--dataset2-manifest",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed")
    )

    args = parser.parse_args()

    ds1_manifest = read_manifest(
        args.dataset1_manifest
    )

    ds2_manifest = read_manifest(
        args.dataset2_manifest
    )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    detection_counts = build_detection(
        ds1_manifest,
        args.output_root
    )

    geometry_counts = build_geometry(
        ds2_manifest,
        args.output_root
    )

    detection_root = (
        args.output_root / "detection"
    )

    geometry_root = (
        args.output_root / "geometry"
    )

    write_yaml(
        detection_root / "data.yaml",
        detection_root,
        "detect"
    )

    write_yaml(
        geometry_root / "data.yaml",
        geometry_root,
        "segment"
    )

    print_counts(
        "DETECTION DATASET",
        detection_counts
    )

    print_counts(
        "GEOMETRY DATASET",
        geometry_counts
    )

    print("\nDatasets created successfully.")

    print(
        f"\nDetection YAML:\n"
        f"{detection_root / 'data.yaml'}"
    )

    print(
        f"\nGeometry YAML:\n"
        f"{geometry_root / 'data.yaml'}"
    )


if __name__ == "__main__":
    main()