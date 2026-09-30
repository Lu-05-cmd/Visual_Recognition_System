from visual_system.models import BoundingBox, Detection
from visual_system.services.tracker import CentroidTracker


def test_outline_stays_unknown_until_shape_is_confirmed():
    tracker = CentroidTracker(max_distance=40)
    first = Detection(
        "product", 0.95, BoundingBox(20, 20, 40, 40), shape="square"
    )
    second = Detection(
        "product", 0.95, BoundingBox(20, 20, 40, 40), shape="triangle"
    )

    tracker.update([first])
    tracker.update([second])

    assert first.shape == "unknown"
    assert second.shape == "unknown"
    assert second.solid_shape == "unknown"


def test_outline_and_solid_shapes_are_confirmed_independently():
    tracker = CentroidTracker(max_distance=40)
    first = Detection("product", 0.95, BoundingBox(20, 20, 40, 40), shape="triangle")
    second = Detection("product", 0.95, BoundingBox(20, 20, 40, 40), shape="rectangle", solid_shape="cylinder")
    tracker.update([first])
    tracker.update([second])
    third = Detection(
        "product",
        0.95,
        BoundingBox(20, 20, 40, 40),
        shape="rectangle",
        solid_shape="cylinder",
    )

    fourth = Detection(
        "product", 0.95, BoundingBox(20, 20, 40, 40), shape="rectangle", solid_shape="cylinder"
    )
    tracker.update([third])
    tracker.update([fourth])

    assert fourth.shape == "rectangle"
    assert fourth.solid_shape == "cylinder"


def test_confirmed_shape_does_not_change_on_one_frame_glitch():
    tracker = CentroidTracker(max_distance=40)
    frames = [
        Detection("product", 0.95, BoundingBox(20, 20, 40, 40), shape=shape)
        for shape in ("square", "square", "square", "triangle", "square")
    ]
    for detection in frames:
        tracker.update([detection])

    assert [item.shape for item in frames] == [
        "unknown", "unknown", "square", "square", "square"
    ]


def test_unknown_evidence_breaks_pending_shape_change_sequence():
    tracker = CentroidTracker(max_distance=40)
    observations = ("square", "square", "square", "circle", "circle", "unknown", "circle")
    frames = [
        Detection("product", 0.95, BoundingBox(20, 20, 40, 40), shape=shape)
        for shape in observations
    ]

    for detection in frames:
        tracker.update([detection])

    assert [item.shape for item in frames] == [
        "unknown", "unknown", "square", "square", "square", "square", "square"
    ]
