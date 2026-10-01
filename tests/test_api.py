from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from visual_system.api import create_app
from visual_system.config import Settings
from visual_system.models import Detection


def test_app_factory_uses_isolated_settings_and_storage(tmp_path: Path) -> None:
    video_results = tmp_path / "results"
    settings = Settings(
        _env_file=None,
        app_name="Test Vision",
        environment="test",
        shape_library_path=tmp_path / "library.sqlite3",
        neural_shape_model_path=tmp_path / "missing-model.pt",
    )
    app = create_app(settings, video_results=video_results)
    client = TestClient(app)

    assert app.title == "Test Vision"
    assert (tmp_path / "library.sqlite3").is_file()
    assert video_results.is_dir()
    assert client.get("/health").json() == {"status": "ok", "environment": "test"}
    assert client.get("/").status_code == 200
    assert client.get("/api/shapes/library").status_code == 200


def test_app_factory_creates_separate_detectors_per_pipeline(tmp_path: Path) -> None:
    class EmptyDetector:
        def detect(self, frame: np.ndarray) -> list[Detection]:
            return []

    created: list[EmptyDetector] = []

    def detector_factory() -> EmptyDetector:
        detector = EmptyDetector()
        created.append(detector)
        return detector

    settings = Settings(
        _env_file=None,
        shape_library_path=tmp_path / "library.sqlite3",
        neural_shape_model_path=tmp_path / "missing-model.pt",
    )
    app = create_app(
        settings,
        video_results=tmp_path / "results",
        detector_factory=detector_factory,
    )

    assert len(created) == 2
    assert app.state.pipeline.detector is created[0]
    assert app.state.camera_pipeline.detector is created[1]
    assert created[0] is not created[1]
    assert app.state.detector_factory is detector_factory


def test_feature_routers_preserve_existing_api_paths(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        shape_library_path=tmp_path / "library.sqlite3",
        neural_shape_model_path=tmp_path / "missing-model.pt",
    )
    app = create_app(settings, video_results=tmp_path / "results")
    client = TestClient(app)

    expected_paths = {
        "/",
        "/health",
        "/api/reset",
        "/api/camera/frame",
        "/api/video/process",
        "/api/video/result/{filename}",
        "/api/shapes/library",
        "/api/shapes/samples",
        "/api/shapes/samples/{sample_id}",
        "/api/shapes/images/{label}/{filename}",
    }

    assert expected_paths <= set(app.openapi()["paths"])
    assert client.get("/api/video/result/missing.mp4").status_code == 404


def test_shape_sample_upload_rejects_2d_labels(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        shape_library_path=tmp_path / "library.sqlite3",
        neural_shape_model_path=tmp_path / "missing-model.pt",
    )
    client = TestClient(create_app(settings, video_results=tmp_path / "results"))

    response = client.post(
        "/api/shapes/samples",
        data={"label": "circle"},
        files={"files": ("flat.png", b"not-decoded", "image/png")},
    )

    assert response.status_code == 400
    assert "khối 3D" in response.json()["detail"]


def test_shape_sample_upload_rejects_legacy_cube_alias(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        shape_library_path=tmp_path / "library.sqlite3",
        neural_shape_model_path=tmp_path / "missing-model.pt",
    )
    client = TestClient(create_app(settings, video_results=tmp_path / "results"))

    response = client.post(
        "/api/shapes/samples",
        data={"label": "cube"},
        files={"files": ("legacy.png", b"not-decoded", "image/png")},
    )

    assert response.status_code == 400


def test_legacy_cube_sample_image_remains_available(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        shape_library_path=tmp_path / "library.sqlite3",
        neural_shape_model_path=tmp_path / "missing-model.pt",
    )
    app = create_app(settings, video_results=tmp_path / "results")
    filename = f"{'a' * 64}.png"
    image_path = app.state.shape_library.image_root / "cube" / filename
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"legacy cube image")
    client = TestClient(app)

    response = client.get(f"/api/shapes/images/cube/{filename}")

    assert response.status_code == 200
    assert response.content == b"legacy cube image"


def test_reset_only_removes_results_from_its_app(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        shape_library_path=tmp_path / "library.sqlite3",
        neural_shape_model_path=tmp_path / "missing-model.pt",
    )
    results_path = tmp_path / "results"
    results_path.mkdir()
    result = results_path / "one-annotated.mp4"
    unrelated = results_path / "keep.txt"
    result.touch()
    unrelated.touch()
    client = TestClient(create_app(settings, video_results=results_path))

    assert client.post("/api/reset").json() == {"status": "reset"}
    assert not result.exists()
    assert unrelated.is_file()


def test_camera_endpoint_rejects_invalid_image(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        shape_library_path=tmp_path / "library.sqlite3",
        neural_shape_model_path=tmp_path / "missing-model.pt",
    )
    client = TestClient(create_app(settings, video_results=tmp_path / "results"))

    response = client.post(
        "/api/camera/frame",
        files={"file": ("frame.png", np.zeros(16, dtype=np.uint8).tobytes(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Frame camera không hợp lệ"
