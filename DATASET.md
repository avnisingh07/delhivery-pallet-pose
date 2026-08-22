# Section 1 — Dataset & Detection Model

## 1. Objective

Section 1 develops the visual perception components required for pallet
detection and downstream pallet pose estimation.

Two complementary datasets were used:

1. A pallet detection dataset for locating pallets.
2. A pallet-front segmentation dataset for extracting visible pallet-front
   geometry required by Section 2.

The final outputs are:
- a single-class pallet detector;
- a pallet-front segmentation model;
- a deterministic held-out evaluation split;
- a geometry interface for Section 2.

---

## 2. Dataset Sources

### Dataset 1 — Pallet Detection

Source:

Roboflow Universe, `pallet-ff5lh-fp7tw-cb6ol`, version 1.

The exported dataset originally contains two classes:

- `hole`
- `pallet`

For the final detection task, only the `pallet` class is retained.

The `hole` class is not required for the Section 2 pose interface and was
therefore removed from the final detection training target.

### Dataset 2 — Pallet Geometry

Source:

Roboflow Universe, `plh-c-1-veibf-zjeab`, version 1.

The source dataset contains eight classes:

- `front`
- `hole`
- `hole_left`
- `hole_right`
- `pallet`
- `pallet_front`
- `pallet_pocket`
- `wood`

For the geometry model, only `pallet_front` is retained.

The `pallet_front` segmentation provides the visible front geometry used
to derive the front-edge representation passed to Section 2.

---

## 3. Dataset Counts

### Detection Dataset

Original images:

- 920

Final pallet annotations:

- 10,329

Final classes:

- `pallet`

Split:

| Split | Images | Pallet Instances |
|---|---:|---:|
| Train | 644 | 6,929 |
| Validation | 141 | 1,527 |
| Held-out Test | 135 | 1,873 |
| Total | 920 | 10,329 |

### Geometry Dataset

Original images:

- 412

Final `pallet_front` annotations:

- 278

Final classes:

- `pallet_front`

Split:

| Split | Images | Pallet-front Instances |
|---|---:|---:|
| Train | 288 | 199 |
| Validation | 61 | 42 |
| Held-out Test | 63 | 37 |
| Total | 412 | 278 |

---

## 4. Split Protocol

The supplied Roboflow train/validation/test split was not directly used
as the final scientific evaluation split.

A deterministic source-group-aware split was constructed using seed `42`.

Target proportions:

- 70% train
- 15% validation
- 15% held-out test

Images belonging to the same derived source group were assigned to the
same split.

This was done to reduce the risk of source/augmentation variants crossing
the evaluation boundary.

Dataset 1 produced:

- 378 source groups
- 264 train groups
- 56 validation groups
- 58 test groups

Dataset 2 produced:

- 412 derived groups
- 288 train groups
- 61 validation groups
- 63 test groups

The Dataset 2 grouping is based on the exported filenames and therefore
does not guarantee complete independence of the underlying original
photographic scenes.

---

## 5. Annotation Format

The source datasets were exported in YOLO-compatible format.

Dataset 1 originally contained bounding-box annotations.

Dataset 2 contains polygon segmentation annotations.

The processed datasets were converted into single-class YOLO datasets:

### Detection

```text
0 = pallet