from collections import Counter

import cv2
import numpy as np

from visual_system.models import BoundingBox, Detection, FrameResult, InspectionStatus
from visual_system.services import video_processor as video_processor_module
from visual_system.services.video_processor import VideoProcessor


def test_video_processor_writes_annotated_video(tmp_path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (320, 240),
    )
    for offset in (0, 8):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (40 + offset, 70), (130 + offset, 160), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()

    summary = VideoProcessor(100, 80.0, 10).process(input_path, output_path)

    assert output_path.is_file()
    assert summary.processed_frames == 2
    assert summary.total_detections == 1
    assert summary.frame_detections == 2
    assert summary.unique_tracks == 1
    assert summary.colors == {"red": 1}


def test_video_summary_does_not_count_solid_projections_as_flat_shapes(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "solid-shapes.mp4"
    output_path = tmp_path / "solid-shapes-annotated.mp4"
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (160, 120),
    )
    for index in range(6):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + index * 15, 20), 3, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    class FixedShapePipeline:
        def __init__(self, *args, **kwargs) -> None:
            self.frame_id = 0

        def process(self, frame: np.ndarray):
            self.frame_id += 1
            detections = [
                Detection(
                    "product",
                    0.95,
                    BoundingBox(10, 10, 40, 40),
                    color="red",
                    shape="circle",
                    solid_shape="sphere",
                    track_id=1,
                    inspection=InspectionStatus.PASS,
                ),
                Detection(
                    "product",
                    0.95,
                    BoundingBox(80, 10, 40, 40),
                    color="blue",
                    shape="square",
                    solid_shape="box",
                    track_id=2,
                    inspection=InspectionStatus.PASS,
                ),
            ]
            return FrameResult(self.frame_id, frame.shape[1], frame.shape[0], detections)

    monkeypatch.setattr(video_processor_module, "VisionPipeline", FixedShapePipeline)

    summary = VideoProcessor(100, 80.0, 10).process(input_path, output_path)

    assert summary.unique_solid_shapes == {"sphere": 1, "box": 1}
    assert summary.unique_outline_shapes == {}
    assert summary.unique_shapes == {"sphere": 1, "box": 1}


def test_video_processor_passes_neural_checkpoint_to_frame_pipeline(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "neural-input.mp4"
    output_path = tmp_path / "neural-output.mp4"
    writer = cv2.VideoWriter(
        str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (80, 60)
    )
    writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
    writer.release()
    observed = {}
    real_pipeline = video_processor_module.VisionPipeline

    def pipeline_factory(*args, **kwargs):
        observed.update(kwargs)
        return real_pipeline(*args, **kwargs)

    monkeypatch.setattr(video_processor_module, "VisionPipeline", pipeline_factory)
    checkpoint = tmp_path / "shape_net.pt"

    VideoProcessor(100, 80.0, 10, neural_model_path=checkpoint).process(
        input_path, output_path
    )

    assert observed["neural_model_path"] == checkpoint


def test_video_processor_uses_injected_detector_factory(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "detector-input.mp4"
    output_path = tmp_path / "detector-output.mp4"
    writer = cv2.VideoWriter(
        str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (80, 60)
    )
    writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
    writer.release()
    observed = {}

    class EmptyDetector:
        def detect(self, frame: np.ndarray) -> list[Detection]:
            return []

    detector = EmptyDetector()

    def detector_factory() -> EmptyDetector:
        return detector

    real_pipeline = video_processor_module.VisionPipeline

    def pipeline_factory(*args, **kwargs):
        observed.update(kwargs)
        return real_pipeline(*args, **kwargs)

    monkeypatch.setattr(video_processor_module, "VisionPipeline", pipeline_factory)

    VideoProcessor(
        100,
        80.0,
        10,
        detector_factory=detector_factory,
    ).process(input_path, output_path)

    assert observed["detector"] is detector


def test_video_processor_skips_repeated_frames_without_splitting_track(tmp_path) -> None:
    input_path = tmp_path / "repeated-input.mp4"
    output_path = tmp_path / "repeated-annotated.mp4"
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (320, 240),
    )
    for frame_index in range(24):
        frame = np.full((240, 320, 3), 238, dtype=np.uint8)
        offset = max(0, frame_index - 5) * 2
        cv2.rectangle(frame, (50 + offset, 70), (130 + offset, 150), (255, 0, 0), -1)
        if frame_index < 5:
            frame = np.full((240, 320, 3), 238, dtype=np.uint8)
            cv2.rectangle(frame, (50, 70), (130, 150), (255, 0, 0), -1)
        writer.write(frame)
    writer.release()

    summary = VideoProcessor(100, 80.0, 10).process(input_path, output_path)

    assert summary.processed_frames == 24
    assert summary.duplicate_frames_skipped >= 3
    assert summary.unique_tracks == 1
    assert summary.total_detections == 1
    assert summary.frame_detections < summary.processed_frames


def test_annotations_stay_inside_frame_for_objects_at_edges() -> None:
    frame = np.zeros((48, 72, 3), dtype=np.uint8)
    detection = Detection(
        label="product",
        confidence=0.95,
        bounding_box=BoundingBox(58, 38, 14, 10),
        color="red",
        shape="rectangle",
        track_id=1,
        inspection=InspectionStatus.PASS,
    )

    VideoProcessor._draw_annotations(frame, [detection])

    assert np.any(frame[:, -1] != 0)
    assert np.any(frame[-1, :] != 0)


def test_wrapped_label_uses_consistent_readable_text_color() -> None:
    frame = np.zeros((180, 120, 3), dtype=np.uint8)
    VideoProcessor._draw_label(
        frame,
        ("XANH LA | DAT",),
        10,
        90,
        52,
        (35, 105, 25),
        (40, 150, 95),
        below=True,
        text_color=(255, 255, 255),
    )

    panel = frame[90:180, 10:62]
    assert np.count_nonzero(np.all(panel == (35, 105, 25), axis=2)) > 0
    assert np.count_nonzero(np.all(panel == (255, 255, 255), axis=2)) > 0


def test_status_panel_text_contrast_is_independent_from_pass_color() -> None:
    assert VideoProcessor._status_panel_color("pass") != (40, 150, 95)


def test_shape_names_distinguish_solid_types() -> None:
    expected_names = {
        "box": "KHOI HOP",
        "sphere": "KHOI CAU",
        "cylinder": "KHOI TRU",
        "cone": "KHOI NON",
    }

    for solid_shape, shape_name in expected_names.items():
        detection = Detection(
            label="product",
            confidence=0.95,
            bounding_box=BoundingBox(0, 0, 50, 50),
            solid_shape=solid_shape,
        )
        assert VideoProcessor._shape_name(detection) == shape_name


def test_annotation_font_scales_for_high_resolution_frames() -> None:
    small = np.zeros((360, 640, 3), dtype=np.uint8)
    large = np.zeros((1080, 1920, 3), dtype=np.uint8)
    detection = Detection(
        label="product",
        confidence=0.95,
        bounding_box=BoundingBox(200, 200, 300, 300),
        color="blue",
        solid_shape="box",
    )

    VideoProcessor._draw_annotations(small, [detection])
    VideoProcessor._draw_annotations(large, [detection])

    small_label = np.zeros((360, 640, 3), dtype=np.uint8)
    large_label = np.zeros((1080, 1920, 3), dtype=np.uint8)
    VideoProcessor._draw_label(small_label, ("KHOI HOP",), 30, 120, 90, (255, 150, 40), (0, 0, 255))
    VideoProcessor._draw_label(large_label, ("KHOI HOP",), 30, 120, 250, (255, 150, 40), (0, 0, 255))
    small_height = np.count_nonzero(np.any(small_label[:120, 30:280] != 0, axis=2))
    large_height = np.count_nonzero(np.any(large_label[:120, 30:280] != 0, axis=2))
    assert large_height > small_height


def test_solid_annotation_does_not_display_conflicting_outline() -> None:
    detection = Detection(
        label="product",
        confidence=0.95,
        bounding_box=BoundingBox(0, 0, 50, 50),
        shape="triangle",
        solid_shape="sphere",
    )

    assert VideoProcessor._shape_name(detection) == "KHOI CAU"


def test_shape_serial_is_above_color_and_inspection_labels() -> None:
    frame = np.zeros((240, 200, 3), dtype=np.uint8)
    detection = Detection(
        label="product",
        confidence=0.95,
        bounding_box=BoundingBox(40, 80, 100, 100),
        color="green",
        solid_shape="cylinder",
        track_id=7,
        inspection=InspectionStatus.PASS,
    )

    VideoProcessor._draw_annotations(frame, [detection])

    shape_color = VideoProcessor._color_for_detection("green")
    status_color = (40, 150, 95)
    upper_band = frame[50:80, 40:141]
    lower_band = frame[181:211, 40:141]
    assert np.any(np.all(upper_band == shape_color, axis=2))
    assert np.any(np.all(lower_band == VideoProcessor._status_panel_color("pass"), axis=2))


def test_shape_label_prefixes_serial_with_hash() -> None:
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    detection = Detection(
        label="product",
        confidence=0.95,
        bounding_box=BoundingBox(30, 50, 100, 50),
        color="blue",
        solid_shape="box",
        serial_number=12,
    )

    VideoProcessor._draw_annotations(frame, [detection])

    label_region = frame[0:50, 30:130]
    assert np.count_nonzero(label_region) > 0


def test_serial_numbers_start_independently_for_each_color() -> None:
    detections = [
        Detection("product", 0.95, BoundingBox(0, 0, 20, 20), color="yellow", track_id=10),
        Detection("product", 0.95, BoundingBox(0, 0, 20, 20), color="yellow", track_id=11),
        Detection("product", 0.95, BoundingBox(0, 0, 20, 20), color="red", track_id=12),
        Detection("product", 0.95, BoundingBox(0, 0, 20, 20), color="red", track_id=13),
        Detection("product", 0.95, BoundingBox(0, 0, 20, 20), color="yellow", track_id=14),
    ]
    serials_by_track = {}
    color_serial_counts = Counter()

    VideoProcessor._assign_serial_numbers(
        detections,
        serials_by_track,
        color_serial_counts,
    )

    assert [item.serial_number for item in detections] == [1, 2, 1, 2, 3]

    repeated_tracks = [detections[1], detections[2]]
    VideoProcessor._assign_serial_numbers(
        repeated_tracks,
        serials_by_track,
        color_serial_counts,
    )
    assert [item.serial_number for item in repeated_tracks] == [2, 1]

    detections[1].color = "red"
    VideoProcessor._assign_serial_numbers(
        [detections[1]],
        serials_by_track,
        color_serial_counts,
    )
    assert detections[1].serial_number == 2
    assert color_serial_counts == {"yellow": 3, "red": 2}
