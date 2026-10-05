from visual_system.models import BoundingBox, Detection
from visual_system.services.tracker import CentroidTracker


def test_tracker_keeps_the_current_2d_outline_without_promoting_it_to_a_solid():
    tracker = CentroidTracker(max_distance=40)
    first = Detection(
        "product", 0.95, BoundingBox(20, 20, 40, 40), shape="square"
    )
    second = Detection(
        "product", 0.95, BoundingBox(20, 20, 40, 40), shape="triangle"
    )

    tracker.update([first])
    tracker.update([second])

    assert first.shape == "square"
    assert second.shape == "triangle"
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

    assert fourth.shape == "rectangle"
    assert fourth.solid_shape == "cylinder"
    assert fifth.solid_shape == "cylinder"
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
        "unknown", "unknown", "box", "box", "box", "box", "box"
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
        "unknown", "unknown", "box", "box", "box", "box", "box", "box", "box"
    ]
    assert len({item.track_id for item in frames}) == 1


def test_unknown_observations_do_not_reset_valid_shape_evidence():
    tracker = CentroidTracker(max_distance=40)
    observations = ("box", "box", "unknown", "box", "unknown")
    frames = [
        Detection("product", 0.95, BoundingBox(20, 20, 40, 40), solid_shape=shape)
        for shape in observations
    ]

    for detection in frames:
        tracker.update([detection])

    assert [item.shape for item in frames] == ["unknown"] * len(frames)
    assert [item.solid_shape for item in frames] == [
        "unknown", "unknown", "unknown", "box", "box"
    ]


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


def test_tracker_keeps_track_when_color_fluctuates_while_object_moves():
    tracker = CentroidTracker(max_distance=100)
    first = Detection("product", 0.95, BoundingBox(10, 20, 20, 20), color="yellow")
    tracker.update([first])

    moved = Detection("product", 0.95, BoundingBox(90, 20, 20, 20), color="orange")
    tracker.update([moved])

    assert moved.track_id == first.track_id
    assert tracker._next_id == 2


def test_tracker_ignores_brief_color_glitches_and_updates_after_stable_change():
    tracker = CentroidTracker(max_distance=40)
    observations = ("yellow", "orange", "yellow", "orange", "yellow")
    frames = [
        Detection("product", 0.95, BoundingBox(20, 20, 40, 40), color=color)
        for color in observations
    ]

    for detection in frames:
        tracker.update([detection])

    assert [item.color for item in frames] == ["yellow"] * len(frames)

    stable_change = [
        Detection("product", 0.95, BoundingBox(20, 20, 40, 40), color="orange")
        for _ in range(5)
    ]
    for detection in stable_change:
        tracker.update([detection])

    assert [item.color for item in stable_change] == [
        "yellow", "yellow", "orange", "orange", "orange"
    ]
    assert {item.track_id for item in [*frames, *stable_change]} == {1}


def test_default_tracker_keeps_id_across_temporary_detection_loss():
    tracker = CentroidTracker()
    first = Detection("product", 0.95, BoundingBox(40, 30, 30, 30), color="yellow")
    tracker.update([first])

    for _ in range(10):
        tracker.update([])

    recovered = Detection(
        "product",
        0.95,
        BoundingBox(48, 30, 30, 30),
        color="orange",
    )
    tracker.update([recovered])

    assert tracker.max_missed_frames == 20
    assert recovered.track_id == first.track_id
    assert tracker._next_id == 2


def test_tracker_matches_detections_globally_instead_of_in_input_order():
    tracker = CentroidTracker(max_distance=100)
    left = Detection("product", 0.95, BoundingBox(10, 20, 20, 20), color="yellow")
    right = Detection("product", 0.95, BoundingBox(110, 20, 20, 20), color="yellow")
    tracker.update([left, right])

    moved_right = Detection("product", 0.95, BoundingBox(105, 20, 20, 20), color="yellow")
    moved_left = Detection("product", 0.95, BoundingBox(15, 20, 20, 20), color="yellow")
    tracker.update([moved_right, moved_left])

    assert moved_left.track_id == left.track_id
    assert moved_right.track_id == right.track_id
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
        "unknown", "unknown", "unknown", "unknown", "sphere", "sphere", "sphere", "sphere"
    ]
