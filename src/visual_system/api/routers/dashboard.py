from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    dashboard_path = Path(__file__).resolve().parents[2] / "dashboard.html"
    return dashboard_path.read_text(encoding="utf-8")
