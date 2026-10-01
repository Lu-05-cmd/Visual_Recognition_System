from visual_system.models import BoundingBox, Detection
from visual_system.services.tracker import CentroidTracker


def test_tracker_never_confirms_a_2d_outline():
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


def test_tracker_confirms_solid_shape_and_ignores_outline():
    tracker = CentroidTracker(max_distance=40)
    first = Detection("product", 0.95, BoundingBox(20, 20, 40, 40), shape="triangle")
    second = Detection(
        "product",
        0.95,
        BoundingBox(20, 20, 40, 40),
        shape="rectangle",
        solid_shape="cylinder",
    )
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
    fifth = Detection(
        "product", 0.95, BoundingBox(20, 20, 40, 40), solid_shape="cylinder"
    )
    tracker.update([fifth])
    sixth = Detection(
        "product", 0.95, BoundingBox(20, 20, 40, 40), solid_shape="cylinder"
    )
    tracker.update([sixth])

    assert fourth.shape == "unknown"
    assert fourth.solid_shape == "unknown"
    assert fifth.solid_shape == "unknown"
    assert sixth.solid_shape == "cylinder"


def test_confirmed_solid_shape_does_not_change_on_one_frame_glitch():
    tracker = CentroidTracker(max_distance=40)
    frames = [
        Detection("product", 0.95, BoundingBox(20, 20, 40, 40), solid_shape=shape)
        for shape in ("box", "box", "box", "box", "box", "sphere", "box")
    ]
    for detection in frames:
        tracker.update([detection])

    assert [item.shape for item in frames] == ["unknown"] * len(frames)
    assert [item.solid_shape for item in frames] == [
        "unknown", "unknown", "unknown", "unknown", "box", "box", "box"
    ]


def test_confirmed_solid_shape_stays_locked_for_the_track_lifetime():
    tracker = CentroidTracker(max_distance=40)
    observations = ("box", "box", "box", "box", "box", "sphere", "sphere", "sphere", "unknown")
    frames = [
        Detection("product", 0.95, BoundingBox(20, 20, 40, 40), solid_shape=shape)
        for shape in observations
    ]

    for detection in frames:
        tracker.update([detection])

    assert [item.solid_shape for item in frames] == [
        "unknown", "unknown", "unknown", "unknown", "box", "box", "box", "box", "box"
    ]
    assert len({item.track_id for item in frames}) == 1


def test_unknown_evidence_breaks_pending_shape_change_sequence():
    tracker = CentroidTracker(max_distance=40)
    observations = ("box", "box", "box", "unknown", "box", "box", "box", "box")
    frames = [
        Detection("product", 0.95, BoundingBox(20, 20, 40, 40), solid_shape=shape)
        for shape in observations
    ]

    for detection in frames:
        tracker.update([detection])

    assert [item.shape for item in frames] == ["unknown"] * len(frames)
    assert [item.solid_shape for item in frames] == ["unknown"] * len(frames)


def test_tracker_reuses_nearby_track_when_same_color_track_is_out_of_range():
    tracker = CentroidTracker(max_distance=100)
    blue_track = Detection(
        "product", 0.95, BoundingBox(0, 0, 20, 20), color="blue"
    )
    nearby_track = Detection(
        "product", 0.95, BoundingBox(290, 0, 20, 20), color="yellow"
    )
    tracker.update([blue_track, nearby_track])

    moved_detection = Detection(
        "product", 0.95, BoundingBox(270, 0, 20, 20), color="blue"
    )
    tracker.update([moved_detection])

    assert moved_detection.track_id == nearby_track.track_id
    assert moved_detection.track_id != blue_track.track_id
    assert tracker._next_id == 3


def test_initial_wrong_label_is_not_exposed_before_stable_shape_is_confirmed():
    tracker = CentroidTracker(max_distance=40)
    observations = ("box", "sphere", "box", "sphere", "sphere", "sphere", "sphere", "sphere")
    frames = [
        Detection("product", 0.95, BoundingBox(20, 20, 40, 40), solid_shape=shape)
        for shape in observations
    ]

    for detection in frames:
        tracker.update([detection])

    assert [item.solid_shape for item in frames] == [
        "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "sphere"
    ]
