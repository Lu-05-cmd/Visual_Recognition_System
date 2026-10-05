"""Optional convolutional model for 2D outline and 3D solid classification.

PyTorch is imported lazily so the base application has no neural runtime
dependency until a checkpoint is enabled or the training command is used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SIZE = 96
CHECKPOINT_VERSION = 5
MIN_DEPLOYMENT_ACCURACY = 0.70
MIN_TRAIN_SAMPLES_PER_CLASS = 20
MIN_VALIDATION_SAMPLES_PER_CLASS = 5
OUTLINE_CLASSES = (
    "circle", "oval", "triangle", "rectangle", "square", "pentagon", "hexagon", "polygon"
)
SOLID_CLASSES = ("none", "sphere", "cylinder", "cone", "box", "hexagonal_prism")
SOLID_TO_OUTLINE = {
    "sphere": "circle",
    "cylinder": "oval",
    "cone": "triangle",
    "box": "rectangle",
    "hexagonal_prism": "hexagon",
}

SOLID_LABEL_ALIASES = {"cube": "box", "cuboid": "box"}
logger = logging.getLogger(__name__)


def prepare_image(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Make a centered RGB 96x96 tensor source, suppressing the crop background."""
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if mask is not None and mask.shape[:2] == rgb.shape[:2] and cv2.countNonZero(mask) >= 10:
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    else:
        bgr = (
            image
            if image.ndim == 3 and image.shape[2] == 3
            else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        border = np.concatenate((value[0], value[-1], value[:, 0], value[:, -1]))
        difference = cv2.absdiff(value, np.full_like(value, np.median(border)))
        _, binary = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            binary = np.zeros_like(binary)
            cv2.drawContours(binary, [largest], -1, 255, -1)
    points = cv2.findNonZero(binary)
    if points is not None:
        x, y, width, height = cv2.boundingRect(points)
        rgb = rgb[y : y + height, x : x + width]
        binary = binary[y : y + height, x : x + width]
        rgb = np.where(binary[:, :, None] > 0, rgb, 18).astype(np.uint8)
    height, width = rgb.shape[:2]
    scale = (IMAGE_SIZE - 8) / max(height, width)
    resized = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 18, dtype=np.uint8)
    y = (IMAGE_SIZE - resized.shape[0]) // 2
    x = (IMAGE_SIZE - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "Cần cài PyTorch để huấn luyện hoặc chạy neural model: "
            "python -m pip install -e '.[neural]'"
        ) from exc

    class ConvBlock(nn.Sequential):
        def __init__(self, input_channels: int, output_channels: int, pool: bool = True):
            layers: list[Any] = [
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            super().__init__(*layers)

    class ShapeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(3, 24),
                ConvBlock(24, 48),
                ConvBlock(48, 96),
                ConvBlock(96, 128, pool=False),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.25),
            )
            self.outline_head = nn.Linear(128, len(OUTLINE_CLASSES))
            self.solid_head = nn.Linear(128, len(SOLID_CLASSES))

        def forward(self, value):
            features = self.features(value)
            return self.outline_head(features), self.solid_head(features)

    return torch, ShapeNet


class NeuralShapeModel:
    """Loads an optional checkpoint and returns separate outline/solid guesses."""

    def __init__(self, checkpoint_path: Path | str, confidence_threshold: float = 0.85):
        self.checkpoint_path = Path(checkpoint_path)
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._torch = None
        self._load_error: str | None = None
        self._loaded_mtime_ns: int | None = None
        self._attempted_mtime_ns: int | None = None

    @property
    def available(self) -> bool:
        return self.checkpoint_path.is_file()

    def _load(self) -> bool:
        try:
            checkpoint_mtime = self.checkpoint_path.stat().st_mtime_ns
        except OSError:
            self._model = None
            self._torch = None
            self._loaded_mtime_ns = None
            return False
        if self._model is not None and self._loaded_mtime_ns == checkpoint_mtime:
            return True
        if self._attempted_mtime_ns == checkpoint_mtime:
            return False
        self._attempted_mtime_ns = checkpoint_mtime
        try:
            torch, model_type = _torch_modules()
            model = model_type()
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("version") != CHECKPOINT_VERSION:
                raise ValueError("Phiên bản checkpoint neural không được hỗ trợ")
            if tuple(checkpoint.get("outline_classes", ())) != OUTLINE_CLASSES:
                raise ValueError("Danh sách lớp hình dạng 2D không khớp")
            if tuple(checkpoint.get("solid_classes", ())) != SOLID_CLASSES:
                raise ValueError("Danh sách lớp hình khối 3D không khớp")
            report = checkpoint.get("deployment_report") or {}
            enabled_heads = frozenset(report.get("deployable_heads", ()))
            if "solid" not in enabled_heads:
                raise ValueError(
                    f"Checkpoint chưa đạt tiêu chí validation "
                    f"{MIN_DEPLOYMENT_ACCURACY:.0%} cho nhánh khối 3D"
                )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self._torch, self._model = torch, model
            self._loaded_mtime_ns = checkpoint_mtime
            self._load_error = None
            return True
        except Exception as exc:  # Broken/old checkpoint should not stop video analysis.
            self._model = None
            self._torch = None
            self._loaded_mtime_ns = None
            self._load_error = str(exc)
            logger.warning("Neural shape model is unavailable: %s", exc)
            return False

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def predict(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[str, float] | None:
        if not self._load():
            return None
        prepared = prepare_image(image, mask)
        tensor = (
            self._torch.from_numpy(prepared.transpose(2, 0, 1).copy())
            .float()
            .div_(255)
            .unsqueeze(0)
        )
        with self._torch.inference_mode():
            _, solid_logits = self._model(tensor)
            solid = self._torch.softmax(solid_logits[0], dim=0)
        solid_index = int(solid.argmax().item())
        solid_label = SOLID_CLASSES[solid_index]
        solid_confidence = float(solid[solid_index].item())
        if solid_label == "none" or solid_confidence < self.confidence_threshold:
            return "unknown", solid_confidence
        return solid_label, solid_confidence
