import cv2
import numpy as np

from visual_system.models import BoundingBox, Detection


class BackgroundSubtractionDetector:
    """Detect moving objects independently of their color."""

    def __init__(
        self,
        min_area: int = 400,
        max_frame_coverage: float = 0.9,
        history: int = 300,
        var_threshold: float = 32.0,
    ) -> None:
        self.min_area = min_area
        self.max_frame_coverage = max_frame_coverage
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )
        self._initialized = False

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.size == 0:
            return []

        learning_rate = 1.0 if not self._initialized else 0.002
        mask = self._subtractor.apply(frame, learningRate=learning_rate)
        self._initialized = True
        if learning_rate == 1.0:
            return []

        mask = self._clean_mask(mask)
        height, width = frame.shape[:2]
        frame_area = height * width
        max_area = max(1, int(frame_area * self.max_frame_coverage))
        background_hsv = self._border_hsv(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > max_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            box_area = box_width * box_height
            aspect_ratio = max(
                box_width / max(box_height, 1),
                box_height / max(box_width, 1),
            )
            if (
                box_area > max_area
                or aspect_ratio > 3.5
                or area / max(box_area, 1) < 0.35
                or x == 0
                or y == 0
                or x + box_width >= width
                or y + box_height >= height
            ):
                continue
            box = BoundingBox(x, y, box_width, box_height)
            if not self._matches_background_color(frame, box, background_hsv):
                continue
            detections.append(
                Detection(
                    label="moving_product",
                    confidence=0.85,
                    bounding_box=box,
                )
            )
        return sorted(self._merge_regions(detections), key=lambda item: item.bounding_box.x)

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        # MOG2 may split an object and its soft shadow into adjacent blobs.
        mask = np.where(mask == 127, 0, mask).astype(np.uint8)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    @staticmethod
    def _border_hsv(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        border = np.concatenate(
            (
                hsv[0, :, :],
                hsv[max(height - 1, 0), :, :],
                hsv[:, 0, :],
                hsv[:, max(width - 1, 0), :],
            )
        )
        return np.median(border, axis=0)

    @staticmethod
    def _matches_background_color(
        frame: np.ndarray,
        box: BoundingBox,
        background_hsv: np.ndarray,
    ) -> bool:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        crop = hsv[box.y : box.y + box.height, box.x : box.x + box.width]
        if crop.size == 0:
            return False
        sample = np.median(crop.reshape(-1, 3), axis=0)
        hue_distance = abs(float(sample[0]) - float(background_hsv[0]))
        hue_distance = min(hue_distance, 180 - hue_distance)
        saturation_distance = abs(float(sample[1]) - float(background_hsv[1]))
        return hue_distance <= 22 and saturation_distance <= 70

    @staticmethod
    def _merge_regions(detections: list[Detection]) -> list[Detection]:
        """Merge overlapping or nearby pieces belonging to one moving object."""
        regions = [item.bounding_box for item in detections]
        changed = True
        while changed:
            changed = False
            merged: list[BoundingBox] = []
            while regions:
                current = regions.pop(0)
                index = 0
                while index < len(regions):
                    other = regions[index]
                    if BackgroundSubtractionDetector._should_merge(current, other):
                        current = BackgroundSubtractionDetector._union(current, other)
                        regions.pop(index)
                        changed = True
                    else:
                        index += 1
                merged.append(current)
            regions = merged

        return [
            Detection(
                label="moving_product",
                confidence=0.85,
                bounding_box=box,
            )
            for box in regions
        ]

    @staticmethod
    def _should_merge(first: BoundingBox, second: BoundingBox) -> bool:
        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)
        intersection = max(0, right - left) * max(0, bottom - top)
        smaller_area = min(first.width * first.height, second.width * second.height)
        if intersection / max(smaller_area, 1) >= 0.35:
            return True

        horizontal_gap = max(
            first.x - (second.x + second.width),
            second.x - (first.x + first.width),
            0,
        )
        vertical_overlap = max(0, bottom - top)
        return horizontal_gap <= 18 and vertical_overlap >= 0.35 * min(first.height, second.height)

    @staticmethod
    def _union(first: BoundingBox, second: BoundingBox) -> BoundingBox:
        left = min(first.x, second.x)
        top = min(first.y, second.y)
        right = max(first.x + first.width, second.x + second.width)
        bottom = max(first.y + first.height, second.y + second.height)
        return BoundingBox(left, top, right - left, bottom - top)
