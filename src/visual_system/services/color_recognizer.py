import cv2
import numpy as np

from visual_system.models import Detection
from visual_system.services.color_ranges import HSV_COLOR_RANGES


class DominantColorRecognizer:
    min_saturation = 55
    min_value = 35

    def recognize(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        for detection in detections:
            box = detection.bounding_box
            crop = frame[box.y : box.y + box.height, box.x : box.x + box.width]
            crop = self._center_crop(crop)
            detection.color = self._dominant_color(crop)
        return detections

    @staticmethod
    def _center_crop(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return crop
        height, width = crop.shape[:2]
        margin_y = round(height * 0.15)
        margin_x = round(width * 0.15)
        return crop[margin_y : height - margin_y, margin_x : width - margin_x]

    @classmethod
    def _dominant_color(cls, crop: np.ndarray) -> str:
        if crop.size == 0:
            return "unknown"

        blurred = cv2.GaussianBlur(crop, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        color_pixel_counts = {
            color_name: sum(
                cv2.countNonZero(cv2.inRange(hsv, lower, upper))
                for lower, upper in ranges
            )
            for color_name, ranges in HSV_COLOR_RANGES.items()
        }
        color_name, pixel_count = max(
            color_pixel_counts.items(),
            key=lambda item: item[1],
        )
        if pixel_count > 80:
            return color_name

        if float(np.median(val)) < 45:
            return "black"
        if float(np.median(sat)) < 40:
            return "white" if float(np.median(val)) > 180 else "gray"
        return "unknown"
