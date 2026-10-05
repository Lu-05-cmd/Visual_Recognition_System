from collections import Counter

import pytest
from shape_synthetic import FLAT_SHAPES, SOLID_SHAPES, make_scene, scene_box

from visual_system.models import BoundingBox, Detection
from visual_system.services.detector import ColorBlobDetector
from visual_system.services.shape_recognizer import ShapeRecognizer


@pytest.mark.parametrize("shape", FLAT_SHAPES)
def test_flat_silhouettes_use_solid_names_without_exposing_2d_labels(shape: str) -> None:
    expected = {
        "square": "box",
        "hexagon": "hexagonal_prism",
    }.get(shape, "unknown")
    recognizer = ShapeRecognizer()
    for seed in range(12):
        frame = make_scene(shape, seed)
        x1, y1, x2, y2 = scene_box(shape, seed)
        detection = Detection("product", 0.95, BoundingBox(x1, y1, x2-x1, y2-y1), color="blue")
        recognizer.recognize(frame, [detection])
        assert detection.shape == "unknown"
        assert detection.solid_shape == expected


@pytest.mark.parametrize("shape", SOLID_SHAPES)
def test_solid_shape_recognition_across_scene_variations(shape: str) -> None:
    recognizer = ShapeRecognizer()
    outcomes: Counter[str] = Counter()
    errors = []
    for seed in range(12):
        frame = make_scene(shape, seed)
        x1, y1, x2, y2 = scene_box(shape, seed)
        detection = Detection("product", 0.95, BoundingBox(x1, y1, x2-x1, y2-y1), color="blue")
        recognizer.recognize(frame, [detection])
        outcomes[detection.solid_shape] += 1
        if detection.solid_shape != shape:
            errors.append((seed, detection.shape, detection.solid_shape))
    assert not errors, f"{shape}: outcomes={dict(outcomes)}, failures={errors}"


@pytest.mark.parametrize("shape", (*FLAT_SHAPES, *SOLID_SHAPES))
def test_object_detector_localizes_one_synthetic_object(shape: str) -> None:
    detector = ColorBlobDetector(min_area=120)
    failures = []
    for seed in range(8):
        frame = make_scene(shape, seed)
        expected = scene_box(shape, seed)
        detections = detector.detect(frame)
        if not detections:
            failures.append((seed, "missed", 0.0))
            continue
        best_overlap = max(
            _intersection_over_union(
                expected,
                (
                    item.bounding_box.x,
                    item.bounding_box.y,
                    item.bounding_box.x + item.bounding_box.width,
                    item.bounding_box.y + item.bounding_box.height,
                ),
            )
            for item in detections
        )
        if len(detections) != 1 or best_overlap < 0.4:
            failures.append((seed, f"{len(detections)} detections", round(best_overlap, 2)))
    assert not failures, f"{shape}: localization failures {failures}"


def _intersection_over_union(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    first_area = (first[2]-first[0]) * (first[3]-first[1])
    second_area = (second[2]-second[0]) * (second[3]-second[1])
    return intersection / max(first_area + second_area - intersection, 1)
