from __future__ import annotations
from pathlib import Path
import numpy as np

PROMPTS = ["cardboard box", "carton", "package", "box"]

class BoxDetector:
    """
    YOLOE-Seg adapter. No training is performed.
    The model is configured with text prompts once and then used for inference.
    """

    def __init__(self, weights: str = "yoloe-11s-seg.pt", conf: float = 0.30,
                 imgsz: int = 960, device: str | None = None):
        try:
            from ultralytics import YOLOE
        except ImportError as e:
            raise RuntimeError(
                "Install Section 3 dependencies first: pip install -r requirements-section3.txt"
            ) from e

        self.model = YOLOE(weights)
        self.model.set_classes(PROMPTS)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def predict(self, image_path: str) -> list[dict]:
        kwargs = dict(source=image_path, conf=self.conf, imgsz=self.imgsz, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)

        detections = []
        for result in results:
            names = result.names
            boxes = result.boxes
            masks = result.masks

            if boxes is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            polygons = None
            if masks is not None:
                polygons = masks.xy

            for i in range(len(xyxy)):
                cls_id = int(classes[i])
                label = names[cls_id] if isinstance(names, dict) else PROMPTS[cls_id]
                mask = polygons[i].tolist() if polygons is not None else None
                detections.append({
                    "label": label,
                    "confidence": float(confs[i]),
                    "bbox_xyxy": xyxy[i].tolist(),
                    "mask_xy": mask,
                })
        return detections
