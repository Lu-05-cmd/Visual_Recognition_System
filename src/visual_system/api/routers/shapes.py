import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from visual_system.services.shape_library import SHAPE_LABELS

router = APIRouter()


@router.get("/api/shapes/library")
def get_shape_library(request: Request) -> dict:
    return request.app.state.shape_library.summary()


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
    label = request.app.state.shape_library.delete_sample(sample_id)
    if label is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh mẫu")
    return {"id": sample_id, "label": label, "deleted": True}


@router.get("/api/shapes/images/{label}/{filename}")
def get_shape_sample_image(
    request: Request,
    label: str,
    filename: str,
) -> FileResponse:
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
