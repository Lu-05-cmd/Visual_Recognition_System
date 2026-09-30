from fastapi import APIRouter, Request

router = APIRouter()


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
