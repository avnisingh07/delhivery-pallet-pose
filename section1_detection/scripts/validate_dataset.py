#!/usr/bin/env python3
"""
Fast validation of YOLO labels.

Supports:
- detection: class x y w h
- segmentation: class x1 y1 x2 y2 ...

Coordinates are expected to be normalized to.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def validate_label(path: Path, n_classes: int, fix_inplace: bool = False) -> list[str]:
    errors = []
    text = path.read_text(errors="replace").strip()
    if not text:
        return errors

    lines = text.splitlines()
    modified = False
    new_lines = []

    for i, line in enumerate(lines, start=1):
        p = line.split()
        if len(p) < 5:
            errors.append(f"{path}:{i}: too few fields")
            new_lines.append(line)
            continue

        try:
            cls = int(float(p[0]))
            coords = [float(x) for x in p[1:]]
        except ValueError:
            errors.append(f"{path}:{i}: non-numeric field")
            new_lines.append(line)
            continue

        if not 0 <= cls < n_classes:
            errors.append(f"{path}:{i}: class {cls} outside [0,{n_classes})")

        # Determine if coordinates are out of bounds
        out_of_bounds = not all(0.0 <= x <= 1.0 for x in coords)
        is_bbox = len(coords) == 4

        if is_bbox:
            if out_of_bounds:
                errors.append(f"{path}:{i}: bbox coordinate outside [0,1]")
        else:
            if len(coords) < 6 or len(coords) % 2:
                errors.append(f"{path}:{i}: invalid polygon coordinate count")
            if out_of_bounds:
                errors.append(f"{path}:{i}: polygon coordinate outside [0,1]")

        # Apply clipping fix if requested and coordinate format is valid
        if out_of_bounds and fix_inplace:
            # Clamp value between 0.0 and 1.0
            fixed_coords = [max(0.0, min(1.0, x)) for x in coords]
            # Convert back to clean string formatting
            coord_str = " ".join(f"{x:.6f}".rstrip('0').rstrip('.') for x in fixed_coords)
            new_lines.append(f"{cls} {coord_str}")
            modified = True
        else:
            new_lines.append(line)

    if modified and fix_inplace:
        path.write_text("\n".join(new_lines) + "\n")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--classes", type=int, required=True)
    ap.add_argument("--fix", action="store_true", help="Automatically clamp coordinates to [0,1]")
    args = ap.parse_args()

    errors = []

    # Filter out files matching known Roboflow / layout metadata naming styles
    files = [
        f for f in sorted(args.labels.rglob("*.txt"))
        if "README" not in f.name and "notes" not in f.name
    ]

    for f in files:
        errors.extend(validate_label(f, args.classes, fix_inplace=args.fix))

    print(f"Checked {len(files)} label files.")

    if args.fix and errors:
        # Re-check to see if the fixes resolved the cleanable errors
        remaining_errors = []
        for f in files:
            remaining_errors.extend(validate_label(f, args.classes, fix_inplace=False))

        fixed_count = len(errors) - len(remaining_errors)
        print(f"Auto-fixed {fixed_count} coordinate alignment errors.")
        errors = remaining_errors

    if errors:
        print(f"Errors remaining: {len(errors)}")
        print("\n".join(errors[:100]))
        raise SystemExit(1)

    print("No label-format errors found.")


if __name__ == "__main__":
    main()
