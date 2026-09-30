import numpy as np

from visual_system.models import FrameResult
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
        shape_library: ShapeFeatureLibrary | None = None,
        neural_model_path: str | None = None,
    ) -> None:
        self.detector = ColorBlobDetector(min_area)
        self.color_recognizer = DominantColorRecognizer()
        neural_model = NeuralShapeModel(neural_model_path) if neural_model_path else None
        self.shape_recognizer = ShapeRecognizer(neural_model)
        self.shape_library = shape_library
        self.tracker = CentroidTracker(max_distance, max_missed_frames)
        self.inspector = BasicInspector()
        self._frame_id = 0

    def process(self, frame: np.ndarray) -> FrameResult:
        masks = self.detector._create_detection_masks(frame)
        detections = self.detector.detect(frame, masks)
        self.color_recognizer.recognize(frame, detections)
        self.shape_recognizer.recognize(frame, detections, masks, self.shape_library)
        self.tracker.update(detections)
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
