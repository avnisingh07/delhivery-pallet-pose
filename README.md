# Delhivery Pallet Pose Estimation & Load Compliance

Computer-vision pipeline for pallet detection, pallet-front geometry extraction,
pose estimation, and load-compliance assessment.

This repository implements the Delhivery AI/ML Engineer (Computer Vision)
take-home assignment under the available-data and hardware constraints.

---

## Project Status

| Section | Status |
|---|---|
| Section 1 — Dataset & Detection | Complete |
| Section 2 — Pose Estimation | Pending |
| Section 3 — Load Compliance | Pending |
| Section 4 — Deployment & Robustness | Pending |

---

## 1. Objective

The system is designed to process pallet images and produce the visual
information required for downstream robotic perception:

```text
Input Image
    |
    v
Pallet Detection
    |
    v
Pallet Bounding Box
    |
    v
Pallet-Front Segmentation
    |
    v
Visible Front Geometry
    |
    v
Pose Estimation
    |
    v
Load Compliance