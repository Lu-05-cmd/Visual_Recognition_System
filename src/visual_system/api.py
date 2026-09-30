from pathlib import Path
from tempfile import gettempdir
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from visual_system.config import Settings, get_settings
from visual_system.pipeline import VisionPipeline
from visual_system.schemas import VideoProcessResponse, VideoSummaryResponse
from visual_system.services.shape_library import SHAPE_LABELS, ShapeFeatureLibrary
from visual_system.services.video_processor import VideoProcessor

router = APIRouter()


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


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "environment": request.app.state.settings.environment}


@router.post("/api/reset")
def reset_system(request: Request) -> dict[str, str]:
    request.app.state.pipeline.reset()
    request.app.state.camera_pipeline.reset()
    for result_path in request.app.state.video_results.glob("*-annotated.mp4"):
        result_path.unlink(missing_ok=True)
    return {"status": "reset"}


@router.post("/api/camera/frame")
async def process_camera_frame(request: Request, file: UploadFile = File(...)) -> Response:
    payload = await file.read()
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Frame camera không hợp lệ")

    result = request.app.state.camera_pipeline.process(frame)
    VideoProcessor._draw_annotations(frame, result.detections)
    encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not encoded:
        raise HTTPException(status_code=500, detail="Không thể mã hóa frame camera")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@router.post("/api/video/process", response_model=VideoProcessResponse)
async def process_video(
    request: Request,
    file: UploadFile = File(...),
    color: str | None = None,
) -> VideoProcessResponse:
    normalized_color = color.lower() if color else None
    valid_colors = {
        "all",
        "red",
        "green",
        "blue",
        "orange",
        "yellow",
        "purple",
        "black",
        "white",
        "gray",
    }
    if normalized_color and normalized_color not in valid_colors:
        raise HTTPException(status_code=400, detail="Màu không hợp lệ")

    suffix = Path(file.filename or "input.mp4").suffix.lower() or ".mp4"
    job_id = uuid4().hex
    video_results = request.app.state.video_results
    settings = request.app.state.settings
    shape_library = request.app.state.shape_library
    input_path = video_results / f"{job_id}{suffix}"
    output_path = video_results / f"{job_id}-annotated.mp4"
    processing_started = perf_counter()
    input_path.write_bytes(await file.read())

    try:
        target_color = None if normalized_color in {None, "all"} else normalized_color
        summary = VideoProcessor(
            settings.detection_min_area,
            settings.track_max_distance,
            settings.track_max_missed_frames,
            shape_library,
            settings.neural_shape_model_path,
        ).process(input_path, output_path, target_color=target_color)
    finally:
        input_path.unlink(missing_ok=True)
    return VideoProcessResponse(
        video_url=f"/api/video/result/{output_path.name}",
        summary=VideoSummaryResponse(
            **summary.__dict__,
            processing_time_seconds=round(perf_counter() - processing_started, 3),
        ),
    )


@router.get("/api/video/result/{filename}")
def video_result(request: Request, filename: str) -> FileResponse:
    video_results = request.app.state.video_results
    result_path = video_results / filename
    if not result_path.is_file() or result_path.parent != video_results:
        raise FileNotFoundError("Không tìm thấy video kết quả")
    return FileResponse(
        result_path,
        media_type="video/mp4",
        filename=result_path.name,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@router.get("/api/shapes/library")
def get_shape_library(request: Request) -> dict:
    shape_library = request.app.state.shape_library
    return shape_library.summary()


@router.post("/api/shapes/samples")
async def add_shape_sample(
    request: Request,
    files: list[UploadFile] = File(...),
    label: str = Form(...),
) -> dict:
    shape_library = request.app.state.shape_library
    normalized_label = label.strip().lower()
    if normalized_label not in SHAPE_LABELS:
        raise HTTPException(
            status_code=400,
            detail="Vui lòng chọn một hình dạng có trong danh sách",
        )

    if not files:
        raise HTTPException(status_code=400, detail="Cần chọn ít nhất một ảnh")

    results = []
    failures = []
    added_count = 0
    duplicate_count = 0
    restored_count = 0
    for file in files:
        payload = await file.read()
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            failures.append({"filename": file.filename or "unknown", "error": "Ảnh không hợp lệ"})
            continue
        try:
            added, restored, coordinates = shape_library.add_image(image, normalized_label)
        except ValueError as error:
            failures.append({"filename": file.filename or "unknown", "error": str(error)})
            continue
        if added:
            added_count += 1
        else:
            duplicate_count += 1
            restored_count += int(restored)
        results.append(
            {
                "filename": file.filename or "unknown",
                "added": added,
                "restored": restored,
                "coordinates": coordinates,
            }
        )
    return {
        "label": normalized_label,
        "total": len(files),
        "added_count": added_count,
        "duplicate_count": duplicate_count,
        "restored_count": restored_count,
        "failures": failures,
        "results": results,
        "library": shape_library.summary(),
    }


@router.delete("/api/shapes/samples/{sample_id}")
def delete_shape_sample(request: Request, sample_id: int) -> dict:
    shape_library = request.app.state.shape_library
    label = shape_library.delete_sample(sample_id)
    if label is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh mẫu")
    return {"id": sample_id, "label": label, "deleted": True}


@router.get("/api/shapes/images/{label}/{filename}")
def get_shape_sample_image(request: Request, label: str, filename: str) -> FileResponse:
    if (
        label not in SHAPE_LABELS
        or len(filename) != 68
        or not filename.endswith(".png")
        or not all(character in "0123456789abcdef" for character in filename[:-4])
    ):
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh mẫu")
    image_path = request.app.state.shape_library.image_root / label / filename
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh mẫu")
    return FileResponse(image_path, media_type="image/png")


@router.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")


app = create_app()
