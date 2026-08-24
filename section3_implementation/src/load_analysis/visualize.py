from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

def draw_assessment(image_path: str, assessment: dict, output_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    detections = assessment.get("detections", [])
    for d in detections:
        box = d.get("bbox_xyxy")
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            label = f"{d.get('label','box')} {d.get('confidence',0):.2f}"
            cv2.putText(img, label, (x1, max(20, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    y = 25
    cv2.putText(
        img, f"OVERALL: {assessment['overall_verdict']}",
        (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA
    )
    y += 30

    for r in assessment["sop_checks"]:
        text = f"R{r['rule_id']} {r['name']}: {r['verdict']} ({r['confidence']:.2f})"
        cv2.putText(img, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)
        y += 22

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
