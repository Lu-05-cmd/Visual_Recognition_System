import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from visual_system.models import InspectionStatus
from visual_system.pipeline import VisionPipeline
from visual_system.ports import Detector
from visual_system.services.shape_library import ShapeFeatureLibrary


@dataclass(frozen=True)
class VideoSummary:
    frame_count: int
    processed_frames: int
    duplicate_frames_skipped: int
    total_detections: int
    frame_detections: int
    unique_tracks: int
    passed: int
    failed: int
    colors: dict[str, int]
    unique_colors: dict[str, int]
    unique_shapes: dict[str, int]
    unique_outline_shapes: dict[str, int]
    unique_solid_shapes: dict[str, int]
    fps: float
    width: int
    height: int


class VideoProcessor:
    def __init__(
        self,
        min_area: int,
        max_distance: float,
        max_missed_frames: int,
        shape_library: ShapeFeatureLibrary | None = None,
        neural_model_path: str | Path | None = None,
        detector_factory: Callable[[], Detector] | None = None,
    ) -> None:
        self.pipeline_config = (min_area, max_distance, max_missed_frames)
        self.shape_library = shape_library
        self.neural_model_path = neural_model_path
        self.detector_factory = detector_factory

    def process(
        self,
        input_path: Path,
        output_path: Path,
        target_color: str | None = None,
    ) -> VideoSummary:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Không thể mở file video")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_output_path = output_path.with_name(f"{output_path.stem}-raw.mp4")
        writer = cv2.VideoWriter(
            str(raw_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError("Không thể tạo file video kết quả")

        pipeline = VisionPipeline(
            *self.pipeline_config,
            shape_library=self.shape_library,
            neural_model_path=self.neural_model_path,
            detector=(
                self.detector_factory()
                if self.detector_factory is not None
                else None
            ),
        )
        tracks: set[int] = set()
        serials_by_track: dict[int, int] = {}
        color_serial_counts: Counter[str] = Counter()
        color_votes: dict[int, Counter[str]] = defaultdict(Counter)
        outline_shape_votes: dict[int, Counter[str]] = defaultdict(Counter)
        solid_shape_votes: dict[int, Counter[str]] = defaultdict(Counter)
        inspection_votes: dict[int, Counter[str]] = defaultdict(Counter)
        processed_frames = 0
        duplicate_frames_skipped = 0
        frame_detections = 0
        previous_hash: int | None = None
        previous_color_histogram = None
        previous_annotated_frame = None
        normalized_target = target_color.lower() if target_color else None
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame_hash, color_histogram = self._perceptual_hash(frame)
                if (
                    previous_hash is not None
                    and (frame_hash ^ previous_hash).bit_count() <= 1
                    and previous_color_histogram is not None
                    and cv2.compareHist(
                        color_histogram,
                        previous_color_histogram,
                        cv2.HISTCMP_BHATTACHARYYA,
                    ) <= 0.015
                    and previous_annotated_frame is not None
                ):
                    writer.write(previous_annotated_frame)
                    processed_frames += 1
                    duplicate_frames_skipped += 1
                    continue
                result = pipeline.process(frame.copy())
                filtered = result.detections
                frame_detections += len(filtered)
                if normalized_target:
                    for item in filtered:
                        if item.color != normalized_target:
                            item.inspection = InspectionStatus.FAIL
                            item.defects = [*item.defects, "color_mismatch"]
                self._assign_serial_numbers(
                    filtered,
                    serials_by_track,
                    color_serial_counts,
                )
                self._draw_annotations(frame, filtered)
                if normalized_target:
                    self._draw_color_mismatch_boxes(frame, filtered, normalized_target)
                writer.write(frame)
                previous_hash = frame_hash
                previous_color_histogram = color_histogram
                previous_annotated_frame = frame.copy()
                processed_frames += 1
                for item in filtered:
                    if item.track_id is None:
                        continue
                    tracks.add(item.track_id)
                    color_votes[item.track_id][item.color or "unknown"] += 1
                    outline_shape_votes[item.track_id][item.shape or "unknown"] += 1
                    solid_shape_votes[item.track_id][item.solid_shape or "unknown"] += 1
                    inspection_votes[item.track_id][item.inspection.value] += 1
        finally:
            capture.release()
            writer.release()

        self._encode_for_browser(raw_output_path, output_path)

        unique_colors: Counter[str] = Counter()
        unique_shapes: Counter[str] = Counter()
        unique_outline_shapes: Counter[str] = Counter()
        unique_solid_shapes: Counter[str] = Counter()
        passed = 0
        failed = 0
        for track_id in tracks:
            color = color_votes[track_id].most_common(1)[0][0]
            outline_shape = outline_shape_votes[track_id].most_common(1)[0][0]
            solid_votes = solid_shape_votes[track_id]
            solid_shape = (
                next(
                    (label for label, _ in solid_votes.most_common() if label != "unknown"),
                    "unknown",
                )
            )
            unique_colors[color] += 1
            unique_outline_shapes[outline_shape] += 1
            if solid_shape != "unknown":
                unique_solid_shapes[solid_shape] += 1
            unique_shapes[solid_shape if solid_shape != "unknown" else outline_shape] += 1
            statuses = inspection_votes[track_id]
            if statuses["pass"] > statuses["fail"]:
                passed += 1
            else:
                failed += 1
        return VideoSummary(
            frame_count=frame_count,
            processed_frames=processed_frames,
            duplicate_frames_skipped=duplicate_frames_skipped,
            total_detections=len(tracks),
            frame_detections=frame_detections,
            unique_tracks=len(tracks),
            passed=passed,
            failed=failed,
            colors=dict(unique_colors),
            unique_colors=dict(unique_colors),
            unique_shapes=dict(unique_shapes),
            unique_outline_shapes=dict(unique_outline_shapes),
            unique_solid_shapes=dict(unique_solid_shapes),
            fps=fps,
            width=width,
            height=height,
        )

    @staticmethod
    def _assign_serial_numbers(
        detections,
        serials_by_track: dict[int, int],
        color_serial_counts: Counter[str],
    ) -> None:
        for detection in detections:
            if detection.track_id is None:
                continue
            if detection.track_id not in serials_by_track:
                color_serial_counts[detection.color] += 1
                serials_by_track[detection.track_id] = color_serial_counts[detection.color]
            detection.serial_number = serials_by_track[detection.track_id]

    @staticmethod
    def _perceptual_hash(frame) -> tuple[int, np.ndarray]:
        """Compact luminance signature for consecutive near-identical frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        reduced = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        coefficients = cv2.dct(reduced.astype("float32"))[:8, :8]
        values = coefficients.flatten()[1:]
        median = float(np.median(values))
        signature = 0
        for bit, value in enumerate(values):
            if value > median:
                signature |= 1 << bit
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_histogram = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
        cv2.normalize(color_histogram, color_histogram, alpha=1, norm_type=cv2.NORM_L1)
        return signature, color_histogram

    @staticmethod
    def _encode_for_browser(raw_output_path: Path, output_path: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raw_output_path.replace(output_path)
            return
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(raw_output_path),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            raw_output_path.unlink(missing_ok=True)

    @staticmethod
    def _draw_annotations(frame, detections) -> None:
        frame_height, frame_width = frame.shape[:2]
        color_names = {
            "red": "DO",
            "orange": "CAM",
            "yellow": "VANG",
            "green": "XANH LA",
            "blue": "XANH DUONG",
            "purple": "TIM",
            "black": "DEN",
            "white": "TRANG",
            "gray": "XAM",
        }
        status_names = {"pass": "DAT", "fail": "LOI", "unknown": "CHUA RO"}
        for detection in detections:
            box = detection.bounding_box
            region_color = VideoProcessor._color_for_detection(detection.color)
            x1 = min(max(box.x, 0), max(frame_width - 1, 0))
            y1 = min(max(box.y, 0), max(frame_height - 1, 0))
            x2 = min(max(box.x + box.width, x1), max(frame_width - 1, 0))
            y2 = min(max(box.y + box.height, y1), max(frame_height - 1, 0))
            box_thickness = max(2, round(min(frame_height, frame_width) / 180))
            cv2.rectangle(frame, (x1, y1), (x2, y2), region_color, box_thickness)
            status_color = (
                (40, 150, 95)
                if detection.inspection.value == "pass"
                else (0, 0, 255)
            )
            box_width = x2 - x1 + 1
            VideoProcessor._draw_label(
                frame,
                (
                    f"{VideoProcessor._shape_name(detection)} - "
                    f"#{detection.serial_number or detection.track_id or 0}",
                ),
                x1,
                y1,
                box_width,
                region_color,
                status_color,
                text_color=(18, 30, 30),
            )
            VideoProcessor._draw_label(
                frame,
                (
                    f"{color_names.get(detection.color, 'MAU')} | "
                    f"{status_names.get(detection.inspection.value, 'CHUA RO')}",
                ),
                x1,
                y2 + 1,
                box_width,
                VideoProcessor._status_panel_color(detection.inspection.value),
                status_color,
                below=True,
                text_color=(255, 255, 255),
            )

    @staticmethod
    def _draw_color_mismatch_boxes(frame, detections, target_color: str) -> None:
        height, width = frame.shape[:2]
        for detection in detections:
            if detection.color == target_color:
                continue
            box = detection.bounding_box
            x1 = min(max(box.x, 0), max(width - 1, 0))
            y1 = min(max(box.y, 0), max(height - 1, 0))
            x2 = min(max(box.x + box.width, x1), max(width - 1, 0))
            y2 = min(max(box.y + box.height, y1), max(height - 1, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    @staticmethod
    def _shape_name(detection) -> str:
        solid_names = {
            "box": "KHOI HOP",
            "cube": "KHOI HOP",
            "cuboid": "KHOI HOP",
            "sphere": "KHOI CAU",
            "cylinder": "KHOI TRU",
            "cone": "KHOI NON",
        }
        flat_names = {
            "triangle": "TAM GIAC",
            "square": "VUONG",
            "rectangle": "CHU NHAT",
            "pentagon": "NGU GIAC",
            "hexagon": "LUC GIAC",
            "circle": "TRON",
            "oval": "OVAL",
            "polygon": "DA GIAC",
        }
        outline_name = flat_names.get(detection.shape, "CHUA RO")
        solid_name = solid_names.get(detection.solid_shape)
        if not solid_name:
            return outline_name
        return solid_name

    @staticmethod
    def _draw_label(
        frame,
        lines,
        box_x,
        anchor_y,
        box_width,
        background,
        status_color,
        below=False,
        text_color=(18, 30, 30),
        bold=True,
    ) -> None:
        frame_height, frame_width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        minimum_label_width = 96 if below else 72
        available_width = min(max(box_width, minimum_label_width), frame_width - box_x)
        # Scale labels with the source video; small fixed labels become
        # unreadable when the video is displayed at full resolution.
        scale = max(0.35, min(1.0, available_width / 110, frame_height / 100))
        thickness = max(2, round(scale * (3 if bold else 2)))
        padding_x = max(4, round(scale * 7))
        padding_y = max(3, round(scale * 5))
        max_text_width = max(1, available_width - 2 * padding_x)
        fitted_lines: list[str] = []
        for line in lines:
            current = ""
            for word in line.split():
                chunks = [word]
                while cv2.getTextSize(chunks[-1], font, scale, thickness)[0][0] > max_text_width:
                    chunk = chunks[-1]
                    split_at = max(1, len(chunk) - 1)
                    while split_at > 1 and cv2.getTextSize(
                        chunk[:split_at], font, scale, thickness
                    )[0][0] > max_text_width:
                        split_at -= 1
                    chunks[-1:] = [chunk[:split_at], chunk[split_at:]]
                for chunk in chunks:
                    candidate = f"{current} {chunk}".strip()
                    if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_text_width:
                        current = candidate
                    else:
                        if current:
                            fitted_lines.append(current)
                        current = chunk
            if current:
                fitted_lines.append(current)

        text_height = cv2.getTextSize("Ag", font, scale, thickness)[0][1]
        line_height = text_height + max(4, round(scale * 6))
        panel_width = available_width
        panel_height = len(fitted_lines) * line_height + 2 * padding_y
        panel_x = box_x
        panel_y = anchor_y if below else anchor_y - panel_height
        panel_y = min(max(panel_y, 0), max(frame_height - panel_height, 0))
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width - 1, panel_y + panel_height - 1),
            background,
            -1,
        )
        for index, line in enumerate(fitted_lines):
            baseline = panel_y + padding_y + (index + 1) * line_height - 3
            cv2.putText(
                frame,
                line,
                (panel_x + padding_x, baseline),
                font,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def _status_panel_color(status: str) -> tuple[int, int, int]:
        if status == "pass":
            return (35, 105, 25)
        if status == "fail":
            return (0, 0, 150)
        return (70, 70, 70)

    @staticmethod
    def _color_for_detection(color_name: str) -> tuple[int, int, int]:
        return {
            "red": (0, 0, 255),
            "orange": (0, 165, 255),
            "yellow": (0, 220, 255),
            "green": (40, 190, 70),
            "blue": (255, 150, 40),
            "purple": (220, 80, 180),
            "black": (170, 170, 170),
            "white": (245, 245, 245),
            "gray": (160, 160, 160),
        }.get(color_name, (240, 180, 40))
