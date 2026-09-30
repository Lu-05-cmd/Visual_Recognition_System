from pathlib import Path
from tempfile import gettempdir

from fastapi import FastAPI

from visual_system.api.routers import router
from visual_system.config import Settings, get_settings
from visual_system.pipeline import VisionPipeline
from visual_system.services.shape_library import ShapeFeatureLibrary


def create_app(
    settings: Settings | None = None,
    *,
    shape_library: ShapeFeatureLibrary | None = None,
    video_results: Path | None = None,
) -> FastAPI:
    settings = settings if settings is not None else get_settings()
    shape_library = (
        shape_library
        if shape_library is not None
        else ShapeFeatureLibrary(settings.shape_library_path)
    )
    video_results = (
        video_results
        if video_results is not None
        else Path(gettempdir()) / "vision-system-results"
    )
    video_results.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.state.settings = settings
    app.state.shape_library = shape_library
    app.state.pipeline = VisionPipeline(
        settings.detection_min_area,
        settings.track_max_distance,
        settings.track_max_missed_frames,
        shape_library,
        settings.neural_shape_model_path,
    )
    app.state.camera_pipeline = VisionPipeline(
        settings.detection_min_area,
        settings.track_max_distance,
        settings.track_max_missed_frames,
        shape_library,
        settings.neural_shape_model_path,
    )
    app.state.video_results = video_results
    app.include_router(router)
    return app
