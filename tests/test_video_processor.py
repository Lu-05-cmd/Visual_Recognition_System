from collections import Counter

import cv2
import numpy as np

from visual_system.models import BoundingBox, Detection, InspectionStatus
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
    assert np.any(np.all(lower_band == status_color, axis=2))


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
