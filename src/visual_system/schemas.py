from pydantic import BaseModel


class DetectionResponse(BaseModel):
    track_id: int | None
    serial_number: int | None
    label: str
    confidence: float
    color: str
    shape: str
    solid_shape: str
    inspection: str
    defects: list[str]
    x: int
    y: int
    width: int
    height: int


class FrameResponse(BaseModel):
    frame_id: int
    width: int
    height: int
    detections: list[DetectionResponse]


class VideoSummaryResponse(BaseModel):
    processing_time_seconds: float
    frame_count: int
    processed_frames: int
    duplicate_frames_skipped: int
    total_detections: int
    frame_detections: int
    unique_tracks: int
    passed: int
    failed: int
    colors: dict[str, int]
    unique_colors: dict[str, int]
    unique_shapes: dict[str, int]
    unique_outline_shapes: dict[str, int]
    unique_solid_shapes: dict[str, int]
    fps: float
    width: int
    height: int


class VideoProcessResponse(BaseModel):
    video_url: str
    summary: VideoSummaryResponse
