import cv2
import numpy as np

from visual_system.models import Detection, InspectionStatus


class BasicInspector:
    def inspect(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        for detection in detections:
            box = detection.bounding_box
            crop = frame[box.y : box.y + box.height, box.x : box.x + box.width]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else np.empty((0, 0))
            edge_ratio = (
                float(np.count_nonzero(cv2.Canny(gray, 80, 160)) / gray.size)
                if gray.size
                else 0.0
            )
            detection.inspection = (
                InspectionStatus.FAIL if edge_ratio > 0.25 else InspectionStatus.PASS
            )
            detection.defects = (
                ["excessive_edges"] if detection.inspection is InspectionStatus.FAIL else []
            )
        return detections
