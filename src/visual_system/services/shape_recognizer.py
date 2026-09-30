import cv2
import numpy as np

from visual_system.models import Detection
from visual_system.services.neural_shape_model import NeuralShapeModel


class ShapeRecognizer:
    def __init__(self, neural_model: NeuralShapeModel | None = None) -> None:
        self.neural_model = neural_model

    def recognize(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        masks: list[np.ndarray] | None = None,
        feature_library=None,
    ) -> list[Detection]:
        for detection in detections:
            box = detection.bounding_box
            crop = frame[box.y : box.y + box.height, box.x : box.x + box.width]
            # A generic foreground mask can join an object to a conveyor belt or
            # its shadow. Segment from this detection's own color instead of
            # selecting whichever full-frame mask has the most pixels in its box.
            extracted = self._extract_contour(crop, detection.color)
            # A detector box may tightly fit the object, so an ordinary object
            # can touch the crop border. Only suppress shape labels when the
            # original image itself cuts the detection off at an outer edge.
            truncated = self._detection_is_truncated(frame.shape, box, extracted)
            detection.shape = (
                "unknown" if truncated else self._classify_extracted(extracted)
            )
            detection.solid_shape = (
                "unknown"
                if truncated
                else self._classify_solid_extracted(crop, extracted)
            )
            neural_result = (
                self.neural_model.predict(
                    crop,
                    extracted[0] if extracted else None,
                )
                if self.neural_model and not truncated
                else None
            )
            if neural_result:
                neural_outline, outline_confidence = neural_result["outline"]
                neural_solid, solid_confidence = neural_result["solid"]
                # The outline and solid labels are independent. Require a strong
                # neural margin before it can replace deterministic contour cues.
                if outline_confidence >= self.neural_model.confidence_threshold:
                    detection.shape = neural_outline
                if (
                    neural_solid != "unknown"
                    and solid_confidence >= self.neural_model.confidence_threshold
                ):
                    detection.solid_shape = neural_solid
            learned_shape = (
                feature_library.classify(
                    extracted,
                    labels={
                        "circle", "oval", "triangle", "rectangle", "square",
                        "pentagon", "hexagon", "polygon",
                    },
                )
                if feature_library
                else None
            )
            # The sample library sees a 2D contour and must never infer depth.
            # It may refine a generic polygon label only.
            if learned_shape and detection.shape == "polygon":
                detection.shape = learned_shape
        return detections

    @staticmethod
    def _mask_touches_edge(mask: np.ndarray) -> bool:
        return bool(
            cv2.countNonZero(mask[0, :]) > mask.shape[1] * 0.2
            or cv2.countNonZero(mask[-1, :]) > mask.shape[1] * 0.2
            or cv2.countNonZero(mask[:, 0]) > mask.shape[0] * 0.2
            or cv2.countNonZero(mask[:, -1]) > mask.shape[0] * 0.2
        )

    @classmethod
    def _detection_is_truncated(cls, frame_shape, box, extracted) -> bool:
        if not extracted or not cls._mask_touches_edge(extracted[0]):
            return False
        mask = extracted[0]
        contour = extracted[1]
        x, y, width, height = cv2.boundingRect(contour)
        frame_height, frame_width = frame_shape[:2]
        edge_contact = (
            (box.x <= 0 and x == 0 and cv2.countNonZero(mask[:, 0]) > mask.shape[0] * 0.65)
            or (box.y <= 0 and y == 0 and cv2.countNonZero(mask[0, :]) > mask.shape[1] * 0.65)
            or (
                box.x + box.width >= frame_width
                and x + width >= mask.shape[1]
                and cv2.countNonZero(mask[:, -1]) > mask.shape[0] * 0.65
            )
            or (
                box.y + box.height >= frame_height
                and y + height >= mask.shape[0]
                and cv2.countNonZero(mask[-1, :]) > mask.shape[1] * 0.65
            )
        )
        if not edge_contact:
            return False
        # Color segmentation can leak a few pixels at the image border due to
        # antialiasing/noise. Treat it as clipped only when most of the contour
        # bbox reaches the border, as with a genuinely cropped object.
        border_span = (
            height if box.x <= 0 and x == 0 and cv2.countNonZero(mask[:, 0]) > mask.shape[0] * 0.65
            else width if box.y <= 0 and y == 0 and cv2.countNonZero(mask[0, :]) > mask.shape[1] * 0.65
            else height if box.x + box.width >= frame_width and x + width >= mask.shape[1] and cv2.countNonZero(mask[:, -1]) > mask.shape[0] * 0.65
            else width if box.y + box.height >= frame_height and y + height >= mask.shape[0] and cv2.countNonZero(mask[-1, :]) > mask.shape[1] * 0.65
            else 0
        )
        corresponding_size = mask.shape[0] if box.x <= 0 or box.x + box.width >= frame_width else mask.shape[1]
        return border_span >= corresponding_size * 0.65

    @classmethod
    def _classify(cls, crop: np.ndarray) -> str:
        extracted = cls._extract_contour(crop)
        return cls._classify_extracted(extracted)

    @classmethod
    def _classify_extracted(cls, extracted) -> str:
        if extracted is None:
            return "unknown"

        _, contour = extracted
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return "unknown"

        hull = cv2.convexHull(contour)
        approximation = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True)
        vertex_count = len(approximation)
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        roundness = cls._ellipse_roundness(contour)
        rotated_width, rotated_height = cv2.minAreaRect(contour)[1]
        rotated_extent = area / max(rotated_width * rotated_height, 1)
        rotated_ratio = max(rotated_width, rotated_height) / max(
            min(rotated_width, rotated_height), 1
        )
        if roundness <= 0.035 and circularity >= 0.68 and rotated_extent <= 0.88:
            return "circle" if rotated_ratio <= 1.15 else "oval"
        if vertex_count == 3:
            return "triangle"
        if vertex_count == 4:
            return "square" if rotated_ratio <= 1.1 else "rectangle"
        if vertex_count == 5:
            return "pentagon"
        if vertex_count == 6:
            return "hexagon"

        return "polygon"

    @staticmethod
    def _ellipse_roundness(contour: np.ndarray) -> float:
        """Normalized radial residual from the best fitting ellipse."""
        if len(contour) < 5:
            return float("inf")
        (center_x, center_y), (axis_width, axis_height), angle = cv2.fitEllipse(contour)
        if min(axis_width, axis_height) <= 0:
            return float("inf")
        points = contour[:, 0, :].astype(np.float32)
        radians = np.deg2rad(angle)
        delta_x = points[:, 0] - center_x
        delta_y = points[:, 1] - center_y
        axis_x = (delta_x * np.cos(radians) + delta_y * np.sin(radians)) / (axis_width / 2)
        axis_y = (-delta_x * np.sin(radians) + delta_y * np.cos(radians)) / (axis_height / 2)
        radial_distance = np.sqrt(axis_x**2 + axis_y**2)
        return float(np.sqrt(np.mean((radial_distance - 1) ** 2)))

    @classmethod
    def _classify_solid(cls, crop: np.ndarray) -> str:
        extracted = cls._extract_contour(crop)
        return cls._classify_solid_extracted(crop, extracted)

    @classmethod
    def _classify_solid_extracted(cls, crop: np.ndarray, extracted) -> str:
        if extracted is None:
            return "unknown"

        mask, contour = extracted
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return "unknown"

        approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertex_count = len(approximation)
        _, _, width, height = cv2.boundingRect(contour)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        rotated_width, rotated_height = cv2.minAreaRect(contour)[1]
        projected_aspect = max(rotated_width, rotated_height) / max(
            min(rotated_width, rotated_height), 1
        )
        circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter)

        object_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, -1)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        values = gray[object_mask > 0]
        if values.size == 0:
            return "unknown"
        shading_range = float(np.percentile(values, 90) - np.percentile(values, 10))
        has_shading = shading_range >= 20 and float(np.std(values)) >= 8
        has_elliptical_rim = cls._has_elliptical_rim(
            gray,
            object_mask,
            cv2.boundingRect(contour),
        )
        object_pixels = crop[object_mask > 0]
        color_spread = float(np.max(np.std(object_pixels, axis=0))) if object_pixels.size else 0.0
        has_internal_facets = cls._has_internal_facets(gray, object_mask, width, height)
        has_cylinder_rim = cls._has_horizontal_rim(gray, object_mask, width, height)
        has_rounded_ends = cls._has_rounded_cylinder_ends(contour, width, height)

        # Uniformly colored silhouettes do not provide depth evidence. Avoid
        # labeling a flat square as a cube from its four contour vertices.
        if (
            color_spread < 2
            and not has_elliptical_rim
            and not (aspect_ratio >= 1.3 and vertex_count >= 5)
        ):
            return "unknown"

        if circularity >= 0.78 and aspect_ratio <= 1.15 and vertex_count >= 7:
            return "sphere" if has_shading else "unknown"
        if vertex_count == 3:
            return (
                "cone"
                if has_internal_facets
                or cls._has_cone_shading(gray, object_mask, cv2.boundingRect(contour))
                else "unknown"
            )
        # A cylinder needs a rounded end cue. A plain elongated rectangle is
        # not enough because box faces and shadows often create the same ratio.
        if aspect_ratio >= 1.15 and (
            has_elliptical_rim or has_cylinder_rim or (has_rounded_ends and not has_internal_facets)
        ):
            return "cylinder"
        if has_internal_facets and vertex_count in {4, 5, 6}:
            return "box"
        return "unknown"

    @staticmethod
    def _has_rounded_cylinder_ends(contour: np.ndarray, width: int, height: int) -> bool:
        """Detect matching rounded caps at both ends of a tall silhouette."""
        if height < width * 1.15 or width < 20:
            return False
        x, y, _, _ = cv2.boundingRect(contour)
        points = contour[:, 0, :]
        normalized_y = (points[:, 1] - y) / max(height, 1)
        top = points[normalized_y <= 0.2]
        bottom = points[normalized_y >= 0.8]
        if len(top) < 5 or len(bottom) < 5:
            return False
        top_span = int(top[:, 0].max() - top[:, 0].min())
        bottom_span = int(bottom[:, 0].max() - bottom[:, 0].min())
        top_center = float((top[:, 0].max() + top[:, 0].min()) / 2)
        bottom_center = float((bottom[:, 0].max() + bottom[:, 0].min()) / 2)
        return (
            top_span >= width * 0.72
            and bottom_span >= width * 0.72
            and abs(top_span - bottom_span) <= width * 0.3
            and abs(top_center - bottom_center) <= width * 0.2
        )

    @staticmethod
    def _has_internal_facets(
        gray: np.ndarray,
        object_mask: np.ndarray,
        width: int,
        height: int,
    ) -> bool:
        """Require a long internal face boundary, not just overall shading."""
        if min(width, height) < 24:
            return False
        interior = cv2.erode(
            object_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        edges = cv2.bitwise_and(cv2.Canny(gray, 45, 130), interior)
        min_line = max(8, round(min(width, height) * 0.16))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(10, min_line // 2),
            minLineLength=min_line,
            maxLineGap=max(3, round(min_line * 0.08)),
        )
        if lines is None:
            return False
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (int(value) for value in line)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            line_length = float(np.hypot(x2 - x1, y2 - y1))
            if 18 <= angle <= 162 and line_length >= min(width, height) * 0.16:
                return True
        return False

    @staticmethod
    def _has_horizontal_rim(
        gray: np.ndarray,
        object_mask: np.ndarray,
        width: int,
        height: int,
    ) -> bool:
        if width < 28 or height < width * 1.15:
            return False
        interior = cv2.erode(
            object_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        edges = cv2.bitwise_and(cv2.Canny(gray, 35, 110), interior)
        minimum_length = max(12, round(width * 0.3))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(8, minimum_length // 2),
            minLineLength=minimum_length,
            maxLineGap=max(3, round(minimum_length * 0.12)),
        )
        if lines is None:
            return False
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (int(value) for value in line)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            relative_y = ((y1 + y2) / 2) / max(height, 1)
            if (angle <= 12 or angle >= 168) and (relative_y <= 0.38 or relative_y >= 0.62):
                return True
        return False

    @staticmethod
    def _has_cone_shading(
        gray: np.ndarray,
        object_mask: np.ndarray,
        bounds: tuple[int, int, int, int],
    ) -> bool:
        """Require a symmetric center-to-edge gradient in a triangular silhouette."""
        _, y, _, height = bounds
        contrasts = []
        asymmetries = []
        for row in np.linspace(y + height * 0.3, y + height * 0.72, 8).astype(int):
            row = min(max(row, 0), gray.shape[0] - 1)
            columns = np.flatnonzero(object_mask[row] > 0)
            if len(columns) < 12:
                continue
            edge_count = max(2, len(columns) // 5)
            left = gray[row, columns[:edge_count]].astype(np.float32)
            right = gray[row, columns[-edge_count:]].astype(np.float32)
            middle_start = int(len(columns) * 0.4)
            middle_end = max(middle_start + 1, int(len(columns) * 0.6))
            center = gray[row, columns[middle_start:middle_end]].astype(np.float32)
            contrasts.append(
                float(np.median(center) - (np.median(left) + np.median(right)) / 2)
            )
            asymmetries.append(abs(float(np.median(left) - np.median(right))))
        if len(contrasts) < 4:
            return False
        median_contrast = float(np.median(contrasts))
        consistent_direction = np.mean(np.sign(contrasts) == np.sign(median_contrast)) >= 0.75
        return (
            abs(median_contrast) >= 12
            and float(np.median(np.abs(contrasts))) >= 10
            and float(np.median(asymmetries)) <= 20
            and consistent_direction
        )

    @staticmethod
    def _has_elliptical_rim(
        gray: np.ndarray,
        object_mask: np.ndarray,
        bounds: tuple[int, int, int, int],
    ) -> bool:
        x, y, width, height = bounds
        object_area = cv2.countNonZero(object_mask)
        object_values = gray[object_mask > 0]
        if object_area == 0 or object_values.size == 0:
            return False

        median = int(np.median(object_values))
        contrast = cv2.absdiff(gray, np.full_like(gray, median))
        intensity_range = float(
            np.percentile(object_values, 90) - np.percentile(object_values, 10)
        )
        threshold = max(10, round(intensity_range * 0.2))
        feature_mask = np.where((contrast >= threshold) & (object_mask > 0), 255, 0).astype(
            np.uint8
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        feature_mask = cv2.morphologyEx(feature_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            feature_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < object_area * 0.025 or area > object_area * 0.4:
                continue

            feature_x, feature_y, feature_width, feature_height = cv2.boundingRect(contour)
            if feature_width < width * 0.25 or feature_height < width * 0.08:
                continue
            if feature_width / max(feature_height, 1) < 1.05:
                continue

            feature_center_y = feature_y + feature_height / 2
            near_end = (
                feature_center_y <= y + height * 0.32
                or feature_center_y >= y + height * 0.68
            )
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
            rim_vertices = len(cv2.approxPolyDP(contour, 0.03 * perimeter, True))
            if near_end and circularity >= 0.55 and rim_vertices >= 6:
                return True
        return False

    @classmethod
    def _extract_contour(
        cls,
        crop: np.ndarray,
        color: str = "unknown",
        foreground_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if foreground_mask is not None and cv2.countNonZero(foreground_mask) >= 10:
            mask = foreground_mask.copy()
        else:
            mask = cls._color_mask(hsv, color)
            if cv2.countNonZero(mask) < 10:
                mask = cls._contrast_mask(hsv[:, :, 2])

        kernel_size = max(3, (min(crop.shape[:2]) // 70) | 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) >= 10]
        if not contours:
            return None
        # A padded detection crop can be slightly off-center; selecting its
        # largest clean component is more stable than the nearest-to-center
        # heuristic, which could pick a highlight or mask speck.
        return mask, max(contours, key=cv2.contourArea)

    @staticmethod
    def _color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
        ranges = {
            "red": ((0, 45, 20, 10, 255, 255), (170, 45, 20, 179, 255, 255)),
            "orange": ((8, 45, 20, 24, 255, 255),),
            "yellow": ((20, 45, 20, 40, 255, 255),),
            "green": ((36, 35, 20, 90, 255, 255),),
            "blue": ((90, 35, 20, 135, 255, 255),),
            "purple": ((130, 35, 20, 169, 255, 255),),
            "black": ((0, 0, 0, 179, 255, 65),),
            "white": ((0, 0, 170, 179, 75, 255),),
            "gray": ((0, 0, 55, 179, 70, 185),),
        }
        selected = ranges.get(color)
        if selected is None:
            return cv2.inRange(hsv, (0, 35, 20), (179, 255, 255))
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for h1, s1, v1, h2, s2, v2 in selected:
            mask |= cv2.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
        return mask

    @staticmethod
    def _contrast_mask(value_channel: np.ndarray) -> np.ndarray:
        height, width = value_channel.shape
        border = np.concatenate(
            (
                value_channel[0, :],
                value_channel[max(height - 1, 0), :],
                value_channel[:, 0],
                value_channel[:, max(width - 1, 0)],
            )
        )
        background = float(np.median(border))
        difference = cv2.absdiff(value_channel, np.full_like(value_channel, background))
        _, mask = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)
        return mask
