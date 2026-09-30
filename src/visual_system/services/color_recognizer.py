import cv2
import numpy as np

from visual_system.models import Detection


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

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        valid = (sat >= cls.min_saturation) & (val >= cls.min_value)
        if not np.any(valid):
            if float(np.median(val)) < 45:
                return "black"
            if float(np.median(sat)) < 40:
                return "white" if float(np.median(val)) > 180 else "gray"
            return "unknown"

        hue_values = hsv[:, :, 0][valid]
        saturation_values = sat[valid].astype(np.float32)
        value_values = val[valid].astype(np.float32)
        # Use the most populated hue range. Shadows can shift the median of a
        # yellow object toward orange even when most of the object is yellow.
        histogram, _ = np.histogram(hue_values, bins=180, range=(0, 180))
        hue = float(np.argmax(histogram))
        saturation = float(np.median(saturation_values))
        value = float(np.median(value_values))
        if value < cls.min_value:
            return "black"
        if saturation < cls.min_saturation:
            return "white" if value > 180 else "gray"
        if hue < 10 or hue >= 170:
            return "red"
        if hue < 22:
            return "orange"
        if hue < 40:
            return "yellow"
        if hue < 90:
            return "green"
        if hue < 140:
            return "blue"
        return "purple"
