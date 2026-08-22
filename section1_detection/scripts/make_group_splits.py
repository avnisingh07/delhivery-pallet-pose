#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import random
import re


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def source_group(filename: str) -> str:
    """
    Remove Roboflow export identifiers and common augmentation suffixes.
    """

    stem = Path(filename).stem

    # Remove Roboflow fingerprint
    stem = re.sub(r"\.rf\.[A-Za-z0-9]+$", "", stem)

    # Remove image extension token sometimes embedded in Roboflow names
    stem = re.sub(
        r"_(jpg|jpeg|png|webp)$",
        "",
        stem,
        flags=re.IGNORECASE
    )

    # Remove common augmentation suffixes
    stem = re.sub(
        r"([_-](aug|flip|flipped|blur|brightness|noise|"
        r"rotation|rotated|crop|cropped|combined|contrast|"
        r"grayscale|gray|saturation))([_-]?\d+)?$",
        "",
        stem,
        flags=re.IGNORECASE
    )

    return stem


def collect_images(dataset_root: Path):

    images = []

    for split in ["train", "valid", "test"]:

        image_dir = dataset_root / split / "images"

        if not image_dir.exists():
            continue

        for image in image_dir.rglob("*"):

            if (
                image.is_file()
                and image.suffix.lower() in IMAGE_EXTENSIONS
            ):
                images.append(
                    {
                        "image": image.resolve(),
                        "original_split": split,
                        "group": source_group(image.name),
                    }
                )

    return images


def create_split(groups, seed):

    rng = random.Random(seed)

    groups = list(groups)
    rng.shuffle(groups)

    n = len(groups)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_groups = set(groups[:n_train])

    val_groups = set(
        groups[n_train:n_train + n_val]
    )

    test_groups = set(
        groups[n_train + n_val:]
    )

    assignments = {}

    for g in train_groups:
        assignments[g] = "train"

    for g in val_groups:
        assignments[g] = "val"

    for g in test_groups:
        assignments[g] = "test"

    return assignments


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--dataset-name",
        required=True
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    images = collect_images(args.dataset_root)

    if not images:
        raise RuntimeError(
            f"No images found under {args.dataset_root}"
        )

    groups = sorted(
        {item["group"] for item in images}
    )

    assignments = create_split(
        groups,
        args.seed
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        args.output,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "dataset",
                "split",
                "group",
                "image",
                "original_split",
            ]
        )

        for item in images:

            writer.writerow(
                [
                    args.dataset_name,
                    assignments[item["group"]],
                    item["group"],
                    str(item["image"]),
                    item["original_split"],
                ]
            )

    print()
    print(f"Dataset: {args.dataset_name}")
    print(f"Images: {len(images)}")
    print(f"Groups: {len(groups)}")

    for split in ["train", "val", "test"]:

        split_images = [
            x for x in images
            if assignments[x["group"]] == split
        ]

        split_groups = {
            x["group"]
            for x in split_images
        }

        print(
            f"{split}: "
            f"{len(split_images)} images, "
            f"{len(split_groups)} groups"
        )

    print()
    print(f"Manifest written to: {args.output}")


if __name__ == "__main__":
    main()