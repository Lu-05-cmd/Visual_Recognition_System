import cv2
import numpy as np

from visual_system.models import BoundingBox, Detection


class ColorBlobDetector:
    """Baseline detector for solid product blobs; replace with YOLO/ONNX later."""

    def __init__(self, min_area: int = 400, max_frame_coverage: float = 0.9) -> None:
        self.min_area = min_area
        self.max_frame_coverage = max_frame_coverage

    def detect(
        self,
        frame: np.ndarray,
        masks: list[np.ndarray] | None = None,
    ) -> list[Detection]:
        if frame.size == 0:
            return []

        height, width = frame.shape[:2]
        frame_area = height * width
        detections: list[Detection] = []
        max_area = max(1, int(frame_area * min(self.max_frame_coverage, 0.65)))
        min_area = min(self.min_area, max(100, round(frame_area * 0.0006)))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detection_masks = masks if masks is not None else self.create_detection_masks(frame)
        for mask_index, mask in enumerate(detection_masks):
            neutral_mask = mask_index == len(detection_masks) - 1
            allowed_area = min(max_area, round(frame_area * 0.3)) if neutral_mask else max_area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > allowed_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                box_area = w * h
                aspect_ratio = max(w / max(h, 1), h / max(w, 1))
                if (
                    box_area > allowed_area
                    or aspect_ratio > 3.5
                ):
                    continue
                object_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(object_mask, [contour], -1, 255, -1)
                object_hue = hsv[:, :, 0][(object_mask > 0) & (hsv[:, :, 1] >= 45)]
                if object_hue.size and float(np.ptp(object_hue)) > 35:
                    # Broad hue bands overlap at their boundaries. Split this
                    # region using its own dominant hue before shape analysis.
                    dominant_hue = int(np.median(object_hue))
                    hue_mask = cv2.inRange(
                        hsv,
                        (max(0, dominant_hue - 12), 45, 25),
                        (min(179, dominant_hue + 12), 255, 255),
                    )
                    local = hue_mask[y : y + h, x : x + w]
                    local_contours, _ = cv2.findContours(
                        local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if local_contours:
                        local_contour = max(local_contours, key=cv2.contourArea)
                        lx, ly, lw, lh = cv2.boundingRect(local_contour)
                        if cv2.contourArea(local_contour) >= min_area and lw * lh < allowed_area:
                            x, y, w, h = x + lx, y + ly, lw, lh
                detections.append(
                    Detection(
                        "product",
                        0.95,
                        self._expand_box(BoundingBox(x, y, w, h), width, height),
                    )
                )
        if not detections:
            detections = self._detect_neutral_objects(
                detection_masks[-1], width, height, min_area
            )
        unique_detections = self._remove_overlapping_regions(detections)
        return sorted(unique_detections, key=lambda item: item.bounding_box.x)

    def _detect_neutral_objects(
        self,
        mask: np.ndarray,
        width: int,
        height: int,
        min_area: int,
    ) -> list[Detection]:
        frame_area = width * height
        max_area = min(
            max(1, int(frame_area * min(self.max_frame_coverage, 0.65))),
            round(frame_area * 0.3),
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            box_area = box_width * box_height
            if box_area > max_area:
                continue
            detections.append(
                Detection(
                    "product",
                    min(0.75, area / max(box_area, 1)),
                    self._expand_box(BoundingBox(x, y, box_width, box_height), width, height),
                )
            )
        return detections

    @staticmethod
    def _expand_box(box: BoundingBox, frame_width: int, frame_height: int) -> BoundingBox:
        # Keep a small safety margin without making the annotation much larger
        # than the segmented object.
        padding_x = max(6, round(box.width * 0.08))
        padding_y = max(6, round(box.height * 0.08))
        x = max(0, box.x - padding_x)
        y = max(0, box.y - padding_y)
        right = min(frame_width, box.x + box.width + padding_x)
        bottom = min(frame_height, box.y + box.height + padding_y)
        return BoundingBox(x, y, right - x, bottom - y)

    @staticmethod
    def _remove_overlapping_regions(detections: list[Detection]) -> list[Detection]:
        """Keep one box per object and remove boxes nested inside another box."""
        ranked = sorted(
            detections,
            key=lambda item: (
                ColorBlobDetector._box_solidity(item),
                item.bounding_box.width * item.bounding_box.height,
            ),
            reverse=True,
        )
        kept: list[Detection] = []
        for candidate in ranked:
            if any(
                ColorBlobDetector._intersection_over_union(
                    candidate.bounding_box,
                    existing.bounding_box,
                )
                >= 0.25
                or ColorBlobDetector._containment_ratio(
                    candidate.bounding_box,
                    existing.bounding_box,
                )
                >= 0.8
                for existing in kept
            ):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _box_solidity(detection: Detection) -> float:
        box = detection.bounding_box
        return min(box.width, box.height) / max(box.width, box.height, 1)

    @staticmethod
    def _intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
        left = max(first.x, second.x)
        top = max(first.y, second.y)
        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)
        intersection = max(0, right - left) * max(0, bottom - top)
        if intersection == 0:
            return 0.0
        union = first.width * first.height + second.width * second.height - intersection
        return intersection / max(union, 1)

    @staticmethod
    def _containment_ratio(inner: BoundingBox, outer: BoundingBox) -> float:
        left = max(inner.x, outer.x)
        top = max(inner.y, outer.y)
        right = min(inner.x + inner.width, outer.x + outer.width)
        bottom = min(inner.y + inner.height, outer.y + outer.height)
        intersection = max(0, right - left) * max(0, bottom - top)
        return intersection / max(inner.width * inner.height, 1)

    @staticmethod
    def create_detection_masks(frame: np.ndarray) -> list[np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_masks: list[np.ndarray] = []
        hue_ranges = (
            ((0, 10), (170, 179)),
            ((10, 22),),
            ((22, 40),),
            ((40, 85),),
            # Keep blue separate from violet; hue 135+ is purple.
            ((90, 140),),
            ((140, 169),),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for ranges in hue_ranges:
            color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower_hue, upper_hue in ranges:
                color_mask |= cv2.inRange(
                    hsv,
                    (lower_hue, 45, 25),
                    (upper_hue, 255, 255),
                )
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_masks.append(cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel))

        foreground_mask = ColorBlobDetector._neutral_foreground_mask(hsv)
        active_color_masks = [mask for mask in color_masks if np.count_nonzero(mask) >= 100]
        if active_color_masks:
            return [*active_color_masks, foreground_mask]
        return [foreground_mask]

    @staticmethod
    def _create_detection_masks(frame: np.ndarray) -> list[np.ndarray]:
        return ColorBlobDetector.create_detection_masks(frame)

    @staticmethod
    def _neutral_foreground_mask(hsv: np.ndarray) -> np.ndarray:
        """Keep neutral foreground components, excluding belt/background connected to edges."""
        foreground = ColorBlobDetector._foreground_mask(hsv)
        neutral = np.where(hsv[:, :, 1] < 45, foreground, 0).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            neutral, connectivity=8
        )
        height, width = neutral.shape
        result = np.zeros_like(neutral)
        for component in range(1, component_count):
            x, y, box_width, box_height, area = stats[component]
            touches_edges = sum(
                (
                    x == 0,
                    y == 0,
                    x + box_width >= width,
                    y + box_height >= height,
                )
            )
            if touches_edges >= 2 or area > height * width * 0.3:
                continue
            result[labels == component] = 255
        return result

    @staticmethod
    def _foreground_mask(hsv: np.ndarray) -> np.ndarray:
        """Build object candidates from contrast to the border, not from color."""
        height, width = hsv.shape[:2]
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
        border_bgr = np.concatenate(
            (bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]), axis=0
        )
        background = np.median(border_bgr, axis=0)
        difference = np.linalg.norm(bgr - background, axis=2)
        border_difference = np.linalg.norm(border_bgr - background, axis=1)
        threshold = max(18.0, float(np.percentile(border_difference, 75)) * 2.5)
        foreground = (
            difference >= threshold
        )
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        mask[foreground] = 255
        kernel_size = max(3, (min(height, width) // 100) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _keep_background_contrast(
        hsv: np.ndarray,
        family_mask: np.ndarray,
    ) -> np.ndarray:
        """Recover same-hue objects by comparing them with the border background."""
        height, width = hsv.shape[:2]
        border = np.concatenate(
            (
                hsv[0, :, :],
                hsv[max(height - 1, 0), :, :],
                hsv[:, 0, :],
                hsv[:, max(width - 1, 0), :],
            )
        )
        background = np.median(border, axis=0)
        hue_distance = np.abs(hsv[:, :, 0].astype(np.float32) - background[0])
        hue_distance = np.minimum(hue_distance, 180 - hue_distance)
        saturation_distance = np.abs(hsv[:, :, 1].astype(np.float32) - background[1])
        value_distance = np.abs(hsv[:, :, 2].astype(np.float32) - background[2])
        contrast = (
            (hue_distance >= 8)
            | (saturation_distance >= 18)
            | (value_distance >= 22)
        )
        contrast_mask = np.zeros_like(family_mask)
        contrast_mask[contrast] = 255
        result = cv2.bitwise_and(family_mask, contrast_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _dominant_border_family(hsv: np.ndarray) -> int | None:
        height, width = hsv.shape[:2]
        border = np.concatenate(
            (
                hsv[0, :, :],
                hsv[max(height - 1, 0), :, :],
                hsv[:, 0, :],
                hsv[:, max(width - 1, 0), :],
            )
        )
        colored = border[(border[:, 1] >= 45) & (border[:, 2] >= 25)]
        if len(colored) < max(20, (2 * width + 2 * height) // 10):
            return None

        hue = float(np.median(colored[:, 0]))
        family_ranges = (10, 25, 40, 85, 135, 170)
        for index, upper_bound in enumerate(family_ranges):
            if hue < upper_bound:
                return index
        return 0
