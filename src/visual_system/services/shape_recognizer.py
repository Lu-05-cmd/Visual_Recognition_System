from typing import TYPE_CHECKING

import cv2
import numpy as np

from visual_system.models import Detection
from visual_system.services.neural_shape_model import NeuralShapeModel

if TYPE_CHECKING:
    from visual_system.services.shape_library import ShapeFeatureLibrary


class ShapeRecognizer:
    def __init__(
        self,
        neural_model: NeuralShapeModel | None = None,
        shape_library: "ShapeFeatureLibrary | None" = None,
    ) -> None:
        self.neural_model = neural_model
        self.shape_library = shape_library

    def recognize(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> list[Detection]:
        for detection in detections:
            box = detection.bounding_box
            crop = frame[box.y : box.y + box.height, box.x : box.x + box.width]
            extracted = self._extract_contour(crop, detection.color)
            truncated = self._detection_is_truncated(frame.shape, box, extracted)
            detection.shape = "unknown"
            heuristic_shape = (
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
            neural_shape = None
            if neural_result:
                neural_solid, solid_confidence = neural_result
                if (
                    neural_solid != "unknown"
                    and solid_confidence >= self.neural_model.confidence_threshold
                ):
                    neural_shape = neural_solid
            box_like_contour = (
                heuristic_shape == "box"
                and extracted is not None
                and 4
                <= len(
                    cv2.approxPolyDP(
                        extracted[1],
                        0.04 * cv2.arcLength(extracted[1], True),
                        True,
                    )
                )
                <= 8
            )
            library_shape = None
            if (
                self.shape_library
                and extracted
                and not truncated
                and (heuristic_shape == "unknown" or box_like_contour)
            ):
                candidate_shape = self.shape_library.classify(
                    extracted,
                    labels={"sphere", "cylinder", "cone", "box", "hexagonal_prism"},
                )
                if not candidate_shape:
                    candidate_shape = self.shape_library.classify_relaxed(
                        extracted,
                        labels={"sphere", "cylinder", "cone", "box", "hexagonal_prism"},
                        relaxed_threshold=0.40,
                    )
                if candidate_shape and not self._is_flat_silhouette_only(extracted, crop):
                    if not box_like_contour:
                        library_shape = candidate_shape
                    elif candidate_shape == "box":
                        library_shape = candidate_shape
                    elif candidate_shape == "hexagonal_prism":
                        _, _, contour_width, contour_height = cv2.boundingRect(
                            extracted[1]
                        )
                        aspect_ratio = max(contour_width, contour_height) / max(
                            min(contour_width, contour_height),
                            1,
                        )
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        object_mask = np.zeros(extracted[0].shape, dtype=np.uint8)
                        cv2.drawContours(object_mask, [extracted[1]], -1, 255, -1)
                        if aspect_ratio >= 1.08 or self._has_hexagonal_prism_facets(
                            gray,
                            object_mask,
                            contour_width,
                            contour_height,
                        ):
                            library_shape = candidate_shape
            if box_like_contour and library_shape:
                detection.solid_shape = library_shape
            else:
                detection.solid_shape = neural_shape or library_shape or heuristic_shape
        confirmed_boxes = [
            detection
            for detection in detections
            if detection.solid_shape == "box" and detection.color != "unknown"
        ]
        for detection in detections:
            if detection.solid_shape != "unknown" or detection.color == "unknown":
                continue
            if any(
                candidate is not detection
                and self._has_similar_box_appearance(candidate, detection)
                for candidate in confirmed_boxes
            ):
                detection.solid_shape = "box"
        return detections

    @staticmethod
    def _is_flat_silhouette_only(extracted, crop: np.ndarray) -> bool:
        """Reject library labels when the object has no internal depth edges."""
        mask, contour = extracted
        object_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, -1)
        interior = cv2.erode(
            object_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        interior_pixels = cv2.countNonZero(interior)
        if interior_pixels < 10:
            return True
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        internal_edges = cv2.bitwise_and(
            cv2.Canny(gray, 45, 130),
            interior,
        )
        edge_ratio = cv2.countNonZero(internal_edges) / interior_pixels
        return edge_ratio < 0.005


    @staticmethod
    def _has_similar_box_appearance(first: Detection, second: Detection) -> bool:
        if first.color != second.color:
            return False
        first_box = first.bounding_box
        second_box = second.bounding_box
        first_aspect = first_box.width / max(first_box.height, 1)
        second_aspect = second_box.width / max(second_box.height, 1)
        if abs(np.log(max(first_aspect, 1e-6) / max(second_aspect, 1e-6))) > 0.2:
            return False
        first_area = first_box.width * first_box.height
        second_area = second_box.width * second_box.height
        area_ratio = first_area / max(second_area, 1)
        return 0.6 <= area_ratio <= 1.67

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
        left_edge = (
            box.x <= 0
            and x == 0
            and cv2.countNonZero(mask[:, 0]) > mask.shape[0] * 0.65
        )
        top_edge = (
            box.y <= 0 and y == 0 and cv2.countNonZero(mask[0, :]) > mask.shape[1] * 0.65
        )
        right_edge = (
            box.x + box.width >= frame_width
            and x + width >= mask.shape[1]
            and cv2.countNonZero(mask[:, -1]) > mask.shape[0] * 0.65
        )
        bottom_edge = (
            box.y + box.height >= frame_height
            and y + height >= mask.shape[0]
            and cv2.countNonZero(mask[-1, :]) > mask.shape[1] * 0.65
        )
        if not any((left_edge, top_edge, right_edge, bottom_edge)):
            return False
        # Ignore small segmentation leaks; only reject a contour that spans
        # most of the original frame edge.
        border_span = height if left_edge or right_edge else width
        corresponding_size = mask.shape[0] if left_edge or right_edge else mask.shape[1]
        return border_span >= corresponding_size * 0.65

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
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        extent = area / max(width * height, 1)
        box_aspect_ratio = width / max(height, 1)

        object_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(object_mask, [contour], -1, 255, -1)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        values = gray[object_mask > 0]
        if values.size == 0:
            return "unknown"
        brightness_values = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2][
            object_mask > 0
        ]
        shading_range = float(
            np.percentile(brightness_values, 90) - np.percentile(brightness_values, 10)
        )
        has_shading = shading_range >= 20 and float(np.std(brightness_values)) >= 8
        outline_shape = cls._classify_outline(contour)
        ellipse_roundness = cls._ellipse_roundness(contour)
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

        # Keep square and hexagonal silhouettes available for their explicit
        # solid-label mappings even when the object has uniform color.
        if (
            color_spread < 2
            and not has_elliptical_rim
            and not (aspect_ratio >= 1.3 and vertex_count >= 5)
        ):
            if outline_shape == "square":
                return "box"
            if outline_shape == "hexagon":
                return "hexagonal_prism"
            return "unknown"

        sphere_like_outline = (
            outline_shape in {"circle", "oval"}
            and aspect_ratio <= 1.7
            and ellipse_roundness <= 0.16
            and not has_internal_facets
        )
        precise_circle = (
            circularity >= 0.83
            and 0.72 <= extent <= 0.81
            and 0.88 <= box_aspect_ratio <= 1.15
        )
        has_radial_shading = cls._has_radial_shading(
            gray,
            object_mask,
            cv2.boundingRect(contour),
        )
        # A rim is a stronger cylinder cue than an oval outer contour is a
        # sphere cue; test it first to avoid confusing short cylinders for spheres.
        if aspect_ratio >= 1.15 and (
            has_elliptical_rim
            or has_cylinder_rim
            or (has_rounded_ends and has_shading and not has_internal_facets)
        ):
            return "cylinder"
        if (
            precise_circle
            and vertex_count >= 7
            and outline_shape != "hexagon"
            and not has_internal_facets
        ) or sphere_like_outline:
            return "sphere" if has_shading or has_radial_shading else "unknown"
        if vertex_count == 3 or outline_shape == "triangle":
            return (
                "cone"
                if has_internal_facets
                or has_shading
                or cls._has_cone_shading(gray, object_mask, cv2.boundingRect(contour))
                else "unknown"
            )
        if vertex_count >= 6 and cls._has_hexagonal_prism_facets(
            gray,
            object_mask,
            width,
            height,
        ):
            return "hexagonal_prism"
        if has_internal_facets:
            return "box"
        if outline_shape == "square" and 0.85 <= aspect_ratio <= 1.2:
            return "box"
        if (vertex_count == 6 or outline_shape == "hexagon") and aspect_ratio <= 1.4:
            return "hexagonal_prism"
        return "unknown"

    @staticmethod
    def _has_radial_shading(
        gray: np.ndarray,
        object_mask: np.ndarray,
        bounds: tuple[int, int, int, int],
    ) -> bool:
        x, y, width, height = bounds
        if min(width, height) < 20:
            return False
        yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
        center_x, center_y = x + width / 2, y + height / 2
        radial = np.sqrt(
            ((xx - center_x) / max(width / 2, 1)) ** 2
            + ((yy - center_y) / max(height / 2, 1)) ** 2
        )
        inside = object_mask > 0
        center_pixels = gray[inside & (radial <= 0.35)]
        rim_pixels = gray[inside & (radial >= 0.65) & (radial <= 0.9)]
        if center_pixels.size < 10 or rim_pixels.size < 10:
            return False
        center_brightness = float(np.median(center_pixels))
        rim_brightness = float(np.median(rim_pixels))
        return abs(center_brightness - rim_brightness) >= 6

    @staticmethod
    def _classify_outline(contour: np.ndarray) -> str:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return "unknown"
        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True)
        if hull_perimeter <= 0:
            return "unknown"
        approximation = cv2.approxPolyDP(hull, 0.04 * hull_perimeter, True)
        vertices = len(approximation)
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
        rotated_width, rotated_height = cv2.minAreaRect(contour)[1]
        width, height = max(rotated_width, rotated_height), min(rotated_width, rotated_height)
        aspect_ratio = width / max(height, 1)
        extent = cv2.contourArea(contour) / max(rotated_width * rotated_height, 1)
        ellipse_roundness = ShapeRecognizer._ellipse_roundness(contour)
        if circularity >= 0.65 and ellipse_roundness <= 0.12 and extent <= 0.9:
            if 1.2 <= aspect_ratio <= 2.5:
                return "oval"
            if vertices >= 7 and aspect_ratio <= 1.15:
                return "circle"
        if vertices == 3:
            return "triangle"
        if vertices == 4:
            return "square" if aspect_ratio <= 1.15 else "rectangle"
        if vertices == 5:
            return "pentagon"
        if vertices == 6:
            return "hexagon"
        return "polygon"

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
    def _has_hexagonal_prism_facets(
        gray: np.ndarray,
        object_mask: np.ndarray,
        width: int,
        height: int,
    ) -> bool:
        if min(width, height) < 24:
            return False
        interior = cv2.erode(
            object_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        edges = cv2.bitwise_and(cv2.Canny(gray, 45, 130), interior)
        minimum_length = max(8, round(min(width, height) * 0.16))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(10, minimum_length // 2),
            minLineLength=minimum_length,
            maxLineGap=max(3, round(minimum_length * 0.08)),
        )
        if lines is None:
            return False

        directions: list[tuple[float, float]] = []
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (int(value) for value in line)
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < minimum_length:
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            for index, (direction, accumulated_length) in enumerate(directions):
                difference = abs(angle - direction)
                if min(difference, 180 - difference) <= 12:
                    directions[index] = (
                        direction,
                        accumulated_length + length,
                    )
                    break
            else:
                directions.append((angle, length))
        return sum(length >= minimum_length for _, length in directions) >= 3

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
