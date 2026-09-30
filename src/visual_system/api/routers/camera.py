import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from visual_system.services.video_processor import VideoProcessor

router = APIRouter()


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
