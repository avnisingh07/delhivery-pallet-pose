from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from section1_detection.scripts.extract_front_edge import extract_front_edge


DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "section1_detection"
    / "models"
    / "pallet_geometry_best.pt"
)


class Section1GeometryAdapter:
    """
    Frozen Section 1 geometry interface for Section 2.

    Input:
        image

    Output per detected pallet:
        pallet_front mask
        G0 = image-left endpoint of extracted front edge
        G1 = image-right endpoint of extracted front edge
        segmentation confidence
    """

    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        conf_threshold=0.25,
        device=None,
    ):
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.device = device

    def predict(self, image):
        """
        Args:
            image: BGR OpenCV image

        Returns:
            list of dictionaries containing Section 1 geometry.
        """

        kwargs = {
            "source": image,
            "conf": self.conf_threshold,
            "verbose": False,
        }

        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)

        if not results:
            return []

        result = results[0]

        if result.masks is None:
            return []

        outputs = []

        image_h, image_w = image.shape[:2]

        for i, polygon in enumerate(result.masks.xy):

            mask = np.zeros(
                (image_h, image_w),
                dtype=np.uint8,
            )

            polygon = np.asarray(polygon, dtype=np.float32)

            polygon = np.round(polygon).astype(np.int32)

            if len(polygon) < 3:
                continue

            cv2.fillPoly(
                mask,
                [polygon],
                1,
            )

            left, right = extract_front_edge(mask)

            confidence = None

            if result.boxes is not None:
                confidence = float(
                    result.boxes.conf[i].item()
                )

            outputs.append(
                {
                    "instance_id": i,
                    "mask": mask,
                    "G0": left,
                    "G1": right,
                    "confidence": confidence,
                }
            )

        return outputs