from fastapi import APIRouter

from visual_system.api.routers.camera import router as camera_router
from visual_system.api.routers.dashboard import router as dashboard_router
from visual_system.api.routers.health import router as health_router
from visual_system.api.routers.shapes import router as shapes_router
from visual_system.api.routers.video import router as video_router

router = APIRouter()
router.include_router(health_router)
router.include_router(camera_router)
router.include_router(video_router)
router.include_router(shapes_router)
router.include_router(dashboard_router)
