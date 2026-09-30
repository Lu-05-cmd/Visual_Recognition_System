import cv2
import numpy as np

from visual_system.models import BoundingBox, Detection
from visual_system.pipeline import VisionPipeline
from visual_system.services.shape_recognizer import ShapeRecognizer
from visual_system.services.tracker import CentroidTracker


def test_pipeline_detects_colors_and_assigns_tracks() -> None:
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 80), (140, 180), (0, 0, 255), -1)
    cv2.circle(frame, (300, 130), 50, (0, 255, 0), -1)

    result = VisionPipeline(min_area=100).process(frame)

    assert len(result.detections) == 2
    assert {item.color for item in result.detections} == {"red", "green"}
    assert {item.shape for item in result.detections} == {"square", "circle"}
    assert {item.solid_shape for item in result.detections} == {"unknown"}
    assert {item.track_id for item in result.detections} == {1, 2}
    assert all(item.inspection.value == "pass" for item in result.detections)


def test_pipeline_detects_neutral_object_against_light_background() -> None:
    frame = np.full((240, 320, 3), 230, dtype=np.uint8)
    cv2.rectangle(frame, (80, 60), (200, 180), (90, 90, 90), -1)

    result = VisionPipeline(min_area=100).process(frame)

    assert len(result.detections) == 1
    assert result.detections[0].shape == "square"


def test_pipeline_detects_edge_objects_on_a_conveyor_without_using_belt_as_shape():
    frame = np.full((185, 888, 3), (10, 14, 15), dtype=np.uint8)
    cv2.rectangle(frame, (0, 85), (887, 151), (72, 76, 77), -1)
    cv2.line(frame, (0, 151), (887, 151), (130, 133, 134), 2)
    cv2.rectangle(frame, (-20, 78), (37, 117), (170, 20, 245), -1)
    cv2.rectangle(frame, (79, 78), (122, 117), (105, 10, 157), -1)
    cv2.circle(frame, (189, 98), 18, (255, 93, 35), -1)
    cv2.fillPoly(frame, [np.array([(290, 70), (315, 111), (265, 111)])], (50, 165, 55))
    cv2.rectangle(frame, (345, 76), (395, 115), (0, 220, 244), -1)
    cv2.rectangle(frame, (455, 76), (495, 115), (0, 140, 220), -1)
    cv2.circle(frame, (555, 96), 19, (25, 35, 222), -1)

    result = VisionPipeline(min_area=400).process(frame)

    assert len(result.detections) == 7
    assert result.detections[0].bounding_box.x == 0
    assert [item.color for item in result.detections] == [
        "purple", "purple", "blue", "green", "yellow", "orange", "red"
    ]
    assert [item.shape for item in result.detections] == [
        "unknown", "rectangle", "circle", "triangle", "rectangle", "square", "circle"
    ]
    assert all(item.solid_shape == "unknown" for item in result.detections)


def test_pipeline_ignores_conveyor_as_object_and_finds_small_purple_object():
    frame = np.full((240, 640, 3), (8, 12, 14), dtype=np.uint8)
    cv2.rectangle(frame, (0, 145), (639, 207), (70, 74, 75), -1)
    cv2.line(frame, (0, 207), (639, 207), (130, 134, 135), 2)
    cv2.rectangle(frame, (240, 160), (290, 194), (20, 20, 220), -1)
    cv2.fillPoly(
        frame,
        [np.array([(290, 160), (310, 150), (310, 184), (290, 194)])],
        (10, 10, 140),
    )
    cv2.fillPoly(
        frame,
        [np.array([(240, 160), (260, 150), (310, 150), (290, 160)])],
        (30, 30, 245),
    )
    cv2.rectangle(frame, (345, 158), (378, 194), (170, 20, 245), -1)
    cv2.rectangle(frame, (365, 158), (378, 194), (100, 10, 140), -1)
    cv2.circle(frame, (430, 176), 11, (170, 20, 245), -1)

    result = VisionPipeline(min_area=400).process(frame)

    assert len(result.detections) == 3
    assert [item.color for item in result.detections] == ["red", "purple", "purple"]
    assert [item.solid_shape for item in result.detections] == [
        "cuboid", "cube", "unknown"
    ]
    assert max(
        item.bounding_box.width * item.bounding_box.height
        for item in result.detections
    ) < frame.shape[0] * frame.shape[1] * 0.3


def test_pipeline_keeps_detection_but_does_not_guess_shape_for_clipped_object():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.fillPoly(frame, [np.array([(159, 20), (159, 100), (120, 100)])], (255, 100, 20))

    result = VisionPipeline(min_area=100).process(frame)

    assert len(result.detections) == 1
    assert result.detections[0].color == "blue"
    assert result.detections[0].shape == "unknown"
    assert result.detections[0].solid_shape == "unknown"


def test_tracker_grows_box_when_later_frames_reveal_more_of_object() -> None:
    tracker = CentroidTracker(max_distance=100)
    first = Detection("product", 0.8, BoundingBox(45, 45, 20, 20), color="yellow")
    tracker.update([first])

    second = Detection("product", 0.8, BoundingBox(30, 30, 50, 50), color="yellow")
    tracker.update([second])

    assert second.track_id == first.track_id
    assert second.bounding_box == BoundingBox(30, 30, 50, 50)

    third = Detection("product", 0.8, BoundingBox(40, 30, 24, 50), color="yellow")
    tracker.update([third])

    assert third.track_id == first.track_id
    assert third.bounding_box.width == 50
    assert third.bounding_box.height == 50


def test_tracker_smooths_box_motion_and_keeps_track_id_during_jitter() -> None:
    tracker = CentroidTracker(max_distance=100)
    first = Detection("product", 0.9, BoundingBox(40, 40, 20, 20), color="purple")
    tracker.update([first])

    second = Detection("product", 0.9, BoundingBox(48, 40, 20, 20), color="purple")
    tracker.update([second])
    third = Detection("product", 0.9, BoundingBox(40, 40, 20, 20), color="purple")
    tracker.update([third])

    assert second.track_id == first.track_id == third.track_id
    assert second.bounding_box.x == 45
    assert third.bounding_box.x == 42

def test_pipeline_ignores_contours_covering_the_whole_frame() -> None:
    frame = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (310, 230), (0, 0, 0), -1)

    result = VisionPipeline(min_area=100).process(frame)

    assert result.detections == []


def test_shape_recognizer_identifies_shaded_sphere() -> None:
    size = 120
    yy, xx = np.mgrid[:size, :size]
    radius = np.sqrt((xx - 60) ** 2 + (yy - 60) ** 2)
    inside = radius <= 44
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[:, :, 0] = 60
    hsv[:, :, 1] = 220
    hsv[:, :, 2] = np.where(inside, 220 - np.minimum(radius * 2.5, 130), 0)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    detection = Detection("product", 0.95, BoundingBox(0, 0, size, size))

    ShapeRecognizer().recognize(frame, [detection])

    assert detection.shape == "circle"
    assert detection.solid_shape == "sphere"


def test_shape_recognizer_identifies_rounded_cylinder_profile() -> None:
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    cv2.rectangle(frame, (38, 42), (82, 118), (0, 0, 255), -1)
    cv2.circle(frame, (60, 42), 22, (0, 0, 255), -1)
    cv2.circle(frame, (60, 118), 22, (0, 0, 255), -1)
    detection = Detection("product", 0.95, BoundingBox(0, 0, 120, 160))

    ShapeRecognizer().recognize(frame, [detection])

    assert detection.solid_shape == "cylinder"


def test_shape_recognizer_uses_elliptical_rim_to_identify_cylinder() -> None:
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    cv2.rectangle(frame, (30, 30), (90, 135), (0, 0, 155), -1)
    cv2.ellipse(frame, (60, 39), (30, 13), 0, 0, 360, (0, 0, 220), -1)
    cv2.ellipse(frame, (60, 39), (28, 9), 0, 0, 180, (0, 0, 85), 3)
    detection = Detection("product", 0.95, BoundingBox(0, 0, 120, 160))

    ShapeRecognizer().recognize(frame, [detection])

    assert detection.solid_shape == "cylinder"


def test_shape_recognizer_detects_cylinder_with_nearly_round_top_rim() -> None:
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    cv2.rectangle(frame, (35, 30), (85, 135), (0, 0, 155), -1)
    cv2.ellipse(frame, (60, 40), (24, 21), 0, 0, 360, (0, 0, 220), -1)
    cv2.ellipse(frame, (60, 40), (22, 18), 0, 0, 180, (0, 0, 85), 3)

    assert _solid_shape(frame) == "cylinder"


def test_shape_recognizer_labels_rounded_rectangular_box_as_cuboid() -> None:
    frame = np.zeros((160, 120, 3), dtype=np.uint8)
    cv2.rectangle(frame, (38, 35), (82, 125), (0, 0, 190), -1)
    cv2.rectangle(frame, (30, 43), (90, 117), (0, 0, 190), -1)
    for center in ((38, 43), (82, 43), (38, 117), (82, 117)):
        cv2.circle(frame, center, 8, (0, 0, 190), -1)
    cv2.rectangle(frame, (61, 35), (90, 125), (0, 0, 95), -1)

    assert _solid_shape(frame) == "cuboid"


def test_shape_recognizer_labels_chamfered_rectangular_box_as_cuboid() -> None:
    frame = np.zeros((170, 120, 3), dtype=np.uint8)
    points = np.array(
        [(38, 25), (82, 25), (92, 35), (92, 135), (82, 145), (38, 145), (28, 135), (28, 35)]
    )
    cv2.fillPoly(frame, [points], (0, 0, 190))
    cv2.fillPoly(
        frame,
        [np.array([(62, 25), (82, 25), (92, 35), (92, 135), (82, 145), (62, 145)])],
        (0, 0, 95),
    )

    assert _solid_shape(frame) == "cuboid"


def test_shape_recognizer_identifies_shaded_polyhedral_solids() -> None:
    cube = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(cube, (30, 30), (90, 90), (0, 0, 190), -1)
    cv2.fillPoly(cube, [np.array([(60, 30), (90, 30), (90, 90), (60, 60)])], (0, 0, 90))

    rectangular_box = np.zeros((100, 140, 3), dtype=np.uint8)
    cv2.rectangle(rectangular_box, (25, 25), (115, 75), (0, 0, 190), -1)
    cv2.fillPoly(
        rectangular_box,
        [np.array([(80, 25), (115, 25), (115, 75), (80, 60)])],
        (0, 0, 90),
    )

    cone = np.zeros((120, 120, 3), dtype=np.uint8)
    cone_points = np.array([(60, 20), (95, 90), (25, 90)])
    cv2.fillPoly(cone, [cone_points], (0, 0, 190))
    cv2.fillPoly(cone, [np.array([(60, 20), (95, 90), (60, 90)])], (0, 0, 90))

    assert _solid_shape(cube) == "cube"
    assert _solid_shape(rectangular_box) == "cuboid"
    assert _solid_shape(cone) == "cone"


def test_shape_recognizer_uses_nearest_colored_object_when_objects_share_frame() -> None:
    frame = np.zeros((180, 360, 3), dtype=np.uint8)
    cv2.circle(frame, (55, 90), 38, (0, 0, 210), -1)
    cv2.rectangle(frame, (125, 45), (195, 135), (0, 0, 210), -1)
    cv2.fillPoly(frame, [np.array([(250, 35), (295, 135), (205, 135)])], (0, 0, 210))
    detections = [
        Detection("product", 0.95, BoundingBox(x, 20, 100, 140), color="red")
        for x in (5, 110, 205)
    ]

    ShapeRecognizer().recognize(frame, detections)

    assert [item.shape for item in detections] == ["circle", "rectangle", "triangle"]


def _solid_shape(frame: np.ndarray) -> str:
    height, width = frame.shape[:2]
    detection = Detection("product", 0.95, BoundingBox(0, 0, width, height))
    ShapeRecognizer().recognize(frame, [detection])
    return detection.solid_shape
