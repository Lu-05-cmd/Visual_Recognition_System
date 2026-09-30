from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from visual_system.schemas import VideoProcessResponse, VideoSummaryResponse
from visual_system.services.video_processor import VideoProcessor

router = APIRouter()
VALID_COLORS = {
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


@router.post("/api/video/process", response_model=VideoProcessResponse)
async def process_video(
    request: Request,
    file: UploadFile = File(...),
    color: str | None = None,
) -> VideoProcessResponse:
    normalized_color = color.lower() if color else None
    if normalized_color and normalized_color not in VALID_COLORS:
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
            request.app.state.detector_factory,
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
        raise HTTPException(status_code=404, detail="Không tìm thấy video kết quả")
    return FileResponse(
        result_path,
        media_type="video/mp4",
        filename=result_path.name,
        headers={"Cache-Control": "no-store, max-age=0"},
    )
