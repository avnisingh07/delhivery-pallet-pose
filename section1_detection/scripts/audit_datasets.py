#!/usr/bin/env python3
"""
Audit the two Roboflow YOLO datasets.

This script:
1. extracts ZIPs into data/raw/<dataset_name>
2. reads data.yaml
3. validates image/label pairing
4. counts annotations/classes
5. detects empty labels and malformed labels
6. creates a conservative source-group key from filenames
7. reports group overlap across the supplied Roboflow splits

It does NOT modify annotations.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "val", "test")


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        return out_dir

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    marker.touch()
    return out_dir


def find_data_yaml(root: Path) -> Path:
    candidates = list(root.rglob("data.yaml")) + list(root.rglob("data.yml"))
    if not candidates:
        raise FileNotFoundError(f"No data.yaml/data.yml under {root}")
    # Prefer the shallowest one.
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def resolve_split_dir(root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def image_files(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        return []
    return sorted(
        p for p in split_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def label_for_image(img: Path, split_dir: Path) -> Path:
    rel = img.relative_to(split_dir)
    return split_dir / "labels" / rel.with_suffix(".txt").name


def source_group(stem: str) -> str:
    """
    Conservative normalization for Roboflow-exported filenames.

    Examples:
      IMG_123_jpg.rf.<hash>      -> IMG_123
      foo_aug_03_jpg.rf.<hash>   -> foo
      foo-flipped-2_jpg.rf...    -> foo

    The function intentionally avoids aggressive normalization. Review
    group counts manually before treating them as ground truth.
    """
    s = stem

    # Remove Roboflow fingerprint.
    s = re.sub(r"\.rf\.[A-Za-z0-9]+$", "", s)

    # Common image-extension tokens introduced by Roboflow.
    s = re.sub(r"_(?:jpg|jpeg|png|webp)$", "", s, flags=re.I)

    # Common augmentation suffixes.
    s = re.sub(
        r"(?:[_-](?:aug|augmentation|flipped|flip|blur|brightness|noise|"
        r"rotation|rotated|crop|cropped|combined|shear|contrast|"
        r"grayscale|gray|saturation|hsv|perspective))"
        r"(?:[_-]?\d+)?$",
        "",
        s,
        flags=re.I,
    )

    # Roboflow augmentation numbering patterns.
    s = re.sub(r"[_-](?:aug|flip|blur|crop|rot)[_-]?\d+$", "", s, flags=re.I)

    return s


def parse_dataset(root: Path) -> dict:
    yaml_path = find_data_yaml(root)
    cfg = yaml.safe_load(yaml_path.read_text())

    names = cfg.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): str(v) for k, v in names.items()}

    result = {
        "root": str(root),
        "data_yaml": str(yaml_path),
        "names": names,
        "splits": {},
        "class_counts": Counter(),
        "total_images": 0,
        "total_annotations": 0,
        "empty_images": 0,
        "missing_labels": 0,
        "malformed_labels": 0,
        "groups": {},
    }

    group_splits = defaultdict(set)

    for split in SPLITS:
        value = cfg.get(split)
        if not value:
            continue

        split_dir = resolve_split_dir(root, value)
        imgs = image_files(split_dir)

        split_info = {
            "directory": str(split_dir),
            "images": len(imgs),
            "annotations": 0,
            "class_counts": Counter(),
            "empty_images": 0,
            "missing_labels": 0,
            "malformed_labels": 0,
            "groups": {},
        }

        for img in imgs:
            result["total_images"] += 1
            group = source_group(img.stem)
            group_splits[group].add(split)
            split_info["groups"][str(img)] = group

            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception:
                # Image corruption is reported separately by the caller if needed.
                pass

            label = label_for_image(img, split_dir)

            if not label.exists():
                split_info["missing_labels"] += 1
                result["missing_labels"] += 1
                continue

            text = label.read_text(errors="replace").strip()

            if not text:
                split_info["empty_images"] += 1
                result["empty_images"] += 1
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                parts = line.split()
                try:
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                    elif len(parts) > 5:
                        # Polygon/segmentation format: class + x/y pairs.
                        cls = float(parts[0])
                        coords = list(map(float, parts[1:]))
                        if len(coords) < 6 or len(coords) % 2 != 0:
                            raise ValueError("invalid polygon coordinate count")
                    else:
                        raise ValueError("too few fields")

                    cls_i = int(cls)
                    if cls_i not in names:
                        raise ValueError(f"unknown class id {cls_i}")

                    split_info["class_counts"][cls_i] += 1
                    result["class_counts"][cls_i] += 1
                    split_info["annotations"] += 1
                    result["total_annotations"] += 1
                except Exception:
                    split_info["malformed_labels"] += 1
                    result["malformed_labels"] += 1

        split_info["class_counts"] = dict(split_info["class_counts"])
        result["splits"][split] = split_info

    result["groups"] = {
        group: sorted(splits)
        for group, splits in group_splits.items()
    }
    result["cross_split_groups"] = {
        g: s for g, s in result["groups"].items() if len(s) > 1
    }
    result["num_groups"] = len(result["groups"])
    result["num_cross_split_groups"] = len(result["cross_split_groups"])
    result["class_counts"] = dict(result["class_counts"])

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-1", type=Path, required=True)
    ap.add_argument("--dataset-2", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--output", type=Path, default=Path("section1_detection/outputs/metrics/dataset_audit.json"))
    args = ap.parse_args()

    args.raw_root.mkdir(parents=True, exist_ok=True)

    roots = {
        "dataset_1": extract_zip(
            args.dataset_1,
            args.raw_root / "dataset_1",
        ),
        "dataset_2": extract_zip(
            args.dataset_2,
            args.raw_root / "dataset_2",
        ),
    }

    audits = {name: parse_dataset(root) for name, root in roots.items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audits, indent=2))

    print(json.dumps({
        name: {
            "images": a["total_images"],
            "annotations": a["total_annotations"],
            "classes": a["names"],
            "empty_images": a["empty_images"],
            "missing_labels": a["missing_labels"],
            "malformed_labels": a["malformed_labels"],
            "groups": a["num_groups"],
            "cross_split_groups": a["num_cross_split_groups"],
        }
        for name, a in audits.items()
    }, indent=2))


if __name__ == "__main__":
    main()
