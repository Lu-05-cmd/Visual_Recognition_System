import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from visual_system.services.neural_shape_model import (
    CHECKPOINT_VERSION,
    IMAGE_SIZE,
    OUTLINE_CLASSES,
    SOLID_CLASSES,
    NeuralShapeModel,
    _torch_modules,
    prepare_image,
)


def test_prepare_image_centers_object_and_uses_fixed_canvas():
    image = np.full((120, 180, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (55, 30), (125, 95), (30, 170, 230), -1)

    result = prepare_image(image)

    assert result.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert result.dtype == np.uint8
    assert np.all(result[0, 0] == 18)
    assert np.mean(result[25:72, 25:72]) > 40


def test_neural_model_without_checkpoint_is_inactive(tmp_path: Path):
    model = NeuralShapeModel(tmp_path / "missing.pt")

    assert not model.available
    assert model.predict(np.zeros((32, 32, 3), dtype=np.uint8)) is None


def test_neural_model_reloads_checkpoint_after_retraining(tmp_path: Path):
    pytest.importorskip("torch")
    torch, model_type = _torch_modules()
    checkpoint_path = tmp_path / "shape_net.pt"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model_state_dict": model_type().state_dict(),
            "outline_classes": list(OUTLINE_CLASSES),
            "solid_classes": list(SOLID_CLASSES),
            "deployment_report": {"deployable_heads": ["outline", "solid"]},
        },
        checkpoint_path,
    )
    model = NeuralShapeModel(checkpoint_path)
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    prediction = model.predict(image)
    assert prediction is not None
    assert prediction[0] in {*SOLID_CLASSES, "unknown"}

    previous_time = checkpoint_path.stat().st_mtime_ns
    torch.save({"version": -1}, checkpoint_path)
    checkpoint_path.touch()
    os.utime(checkpoint_path, ns=(previous_time + 2_000_000, previous_time + 2_000_000))

    assert model.predict(image) is None
    assert "Phiên bản checkpoint" in model.load_error


def test_legacy_checkpoint_is_rejected_instead_of_overriding_heuristics(tmp_path: Path):
    pytest.importorskip("torch")
    torch, model_type = _torch_modules()
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "version": 1,
            "model_state_dict": model_type().state_dict(),
            "outline_classes": list(OUTLINE_CLASSES),
            "solid_classes": list(SOLID_CLASSES),
        },
        checkpoint_path,
    )

    model = NeuralShapeModel(checkpoint_path)

    assert model.predict(np.zeros((40, 40, 3), dtype=np.uint8)) is None
    assert "phiên bản checkpoint" in model.load_error.lower()


def test_checkpoint_without_deployable_solid_head_is_rejected(tmp_path: Path):
    pytest.importorskip("torch")
    torch, model_type = _torch_modules()
    checkpoint_path = tmp_path / "outline-only.pt"
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model_state_dict": model_type().state_dict(),
            "outline_classes": list(OUTLINE_CLASSES),
            "solid_classes": list(SOLID_CLASSES),
            "deployment_report": {"deployable_heads": ["outline"]},
        },
        checkpoint_path,
    )

    model = NeuralShapeModel(checkpoint_path)

    assert model.predict(np.zeros((40, 40, 3), dtype=np.uint8)) is None
    assert "khối 3D" in model.load_error
