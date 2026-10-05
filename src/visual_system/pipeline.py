from copy import deepcopy

import numpy as np

from visual_system.models import Detection, FrameResult
from visual_system.ports import Detector
from visual_system.services.color_recognizer import DominantColorRecognizer
from visual_system.services.detector import ColorBlobDetector
from visual_system.services.inspector import BasicInspector
from visual_system.services.neural_shape_model import NeuralShapeModel
from visual_system.services.shape_library import ShapeFeatureLibrary
from visual_system.services.shape_recognizer import ShapeRecognizer
from visual_system.services.tracker import CentroidTracker


class VisionPipeline:
    def __init__(
        self,
        min_area: int = 400,
        max_distance: float = 120.0,
        max_missed_frames: int = 20,
        neural_model_path: str | None = None,
        detector: Detector | None = None,
        shape_library_path: str | None = None,
    ) -> None:
        self.detector = detector if detector is not None else ColorBlobDetector(min_area)
        self.color_recognizer = DominantColorRecognizer()
        neural_model = NeuralShapeModel(neural_model_path) if neural_model_path else None
        shape_library = (
            ShapeFeatureLibrary(shape_library_path) if shape_library_path else None
        )
        self.shape_recognizer = ShapeRecognizer(neural_model, shape_library)
        self.tracker = CentroidTracker(max_distance, max_missed_frames)
        self.inspector = BasicInspector()
        self._frame_id = 0
        self._last_observations: list[Detection] | None = None

    def process(self, frame: np.ndarray) -> FrameResult:
        detections = self.detector.detect(frame)
        self.color_recognizer.recognize(frame, detections)
        self.shape_recognizer.recognize(frame, detections)
        self._last_observations = deepcopy(detections)
        detections = self.tracker.update(detections)
        self.inspector.inspect(frame, detections)
        self._frame_id += 1
        height, width = frame.shape[:2]
        return FrameResult(self._frame_id, width, height, detections)

    def repeat_last_observation(self, frame: np.ndarray) -> FrameResult | None:
        """Advance temporal tracking for a duplicate frame without rerunning vision."""
        if self._last_observations is None:
            return None
        detections = self.tracker.update(deepcopy(self._last_observations))
        self.inspector.inspect(frame, detections)
        self._frame_id += 1
        height, width = frame.shape[:2]
        return FrameResult(self._frame_id, width, height, detections)

    def reset(self) -> None:
        self.tracker = CentroidTracker(
            self.tracker.max_distance,
            self.tracker.max_missed_frames,
        )
        self._frame_id = 0
        self._last_observations = None
