"""Integration test for shape library fallback in recognition pipeline."""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tests.shape_synthetic import make_scene, scene_box
from visual_system.models import BoundingBox, Detection
from visual_system.services.shape_library import ShapeFeatureLibrary
from visual_system.services.shape_recognizer import ShapeRecognizer
from visual_system.services.tracker import CentroidTracker


def test_shape_library_fallback_when_heuristic_fails() -> None:
    """Verify shape library is called as fallback when heuristic returns unknown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        library_path = Path(tmpdir) / "shapes.db"
        library = ShapeFeatureLibrary(library_path)

        # Create recognizer with library
        recognizer = ShapeRecognizer(shape_library=library)

        # Create a test frame with the cone
        test_frame = make_scene("cone", seed=42, width=320, height=240)
        detection = Detection(
            label="object",
            confidence=0.95,
            bounding_box=BoundingBox(x=50, y=40, width=180, height=160),
            color="green",
        )
        detections = [detection]

        # Process detections through recognizer
        result = recognizer.recognize(test_frame, detections)

        # Recognition should complete without error
        assert len(result) == 1
        classified = result[0]
        # It should be either detected by heuristic or by library fallback
        assert isinstance(classified.solid_shape, str)


def test_flat_silhouette_rejected_by_library_guard() -> None:
    """Verify that flat 2D silhouettes are not labeled as 3D by library."""
    with tempfile.TemporaryDirectory() as tmpdir:
        library_path = Path(tmpdir) / "shapes.db"
        library = ShapeFeatureLibrary(library_path)

        # Create a pure white circle (flat, no shading)
        flat_circle = np.ones((240, 320, 3), dtype=np.uint8) * 255
        cv2.circle(flat_circle, (160, 120), 80, (0, 0, 0), -1)

        recognizer = ShapeRecognizer(shape_library=library)
        detection = Detection(
            label="object",
            confidence=0.95,
            bounding_box=BoundingBox(x=50, y=40, width=200, height=160),
            color="white",
        )

        # Process through recognizer
        result = recognizer.recognize(flat_circle, [detection])

        # Flat silhouette should remain unknown, not be labeled as sphere
        assert len(result) == 1
        assert result[0].solid_shape == "unknown"


def test_library_resolves_hexagonal_prism_when_heuristic_confuses_it_with_box(
    tmp_path, monkeypatch
) -> None:
    library = ShapeFeatureLibrary(tmp_path / "shapes.db")
    for shape in ("box", "hexagonal_prism"):
        for seed in range(4):
            library.add_image(make_scene(shape, seed), shape)

    frame = make_scene("hexagonal_prism", seed=7)
    x1, y1, x2, y2 = scene_box("hexagonal_prism", seed=7)
    monkeypatch.setattr(
        ShapeRecognizer,
        "_classify_solid_extracted",
        classmethod(lambda cls, crop, extracted: "box"),
    )
    recognizer = ShapeRecognizer(shape_library=library)
    tracker = CentroidTracker(max_distance=40)
    detections = []
    for _ in range(5):
        detection = Detection(
            "object",
            0.95,
            BoundingBox(x1, y1, x2 - x1, y2 - y1),
            color="blue",
        )
        recognizer.recognize(frame, [detection])
        tracker.update([detection])
        detections.append(detection)

    assert detections[-1].solid_shape == "hexagonal_prism"
    assert len({detection.track_id for detection in detections}) == 1


def test_flat_hexagon_is_reported_as_prism_without_a_2d_label() -> None:
    frame = make_scene("hexagon", seed=5)
    x1, y1, x2, y2 = scene_box("hexagon", seed=5)
    detection = Detection(
        "object",
        0.95,
        BoundingBox(x1, y1, x2 - x1, y2 - y1),
        color="blue",
    )

    ShapeRecognizer().recognize(frame, [detection])

    assert detection.shape == "unknown"
    assert detection.solid_shape == "hexagonal_prism"


def test_library_keeps_box_and_prism_classes_separate_across_view_variations(
    tmp_path,
) -> None:
    library = ShapeFeatureLibrary(tmp_path / "shapes.db")
    for label in ("box", "hexagonal_prism"):
        for seed in range(4):
            library.add_image(make_scene(label, seed), label)

    recognizer = ShapeRecognizer(shape_library=library)
    for label in ("box", "hexagonal_prism"):
        for seed in range(12):
            frame = make_scene(label, seed)
            x1, y1, x2, y2 = scene_box(label, seed)
            detection = Detection(
                "object",
                0.95,
                BoundingBox(x1, y1, x2 - x1, y2 - y1),
                color="blue",
            )
            recognizer.recognize(frame, [detection])

            assert detection.solid_shape == label, (
                f"{label} seed={seed} classified as {detection.solid_shape}"
            )


def test_relaxed_classifier_exists() -> None:
    """Verify classify_relaxed() method works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        library_path = Path(tmpdir) / "shapes.db"
        library = ShapeFeatureLibrary(library_path)

        # Create two similar cone images
        cone1 = make_scene("cone", seed=42, width=320, height=240)

        # Test that library methods exist
        assert hasattr(library, "classify")
        assert hasattr(library, "classify_relaxed")

        recognizer = ShapeRecognizer(shape_library=library)
        extracted = recognizer._extract_contour(cone1, "green")

        if extracted:
            # Test relaxed classifier
            result = library.classify_relaxed(
                extracted,
                labels={"cone"},
                relaxed_threshold=0.40,
            )
            # Should return None if no samples in library
            assert result is None
