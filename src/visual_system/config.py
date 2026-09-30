from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Vision System"
    environment: str = "development"
    camera_source: str = "demo"
    detection_min_area: int = 400
    track_max_distance: float = 120.0
    track_max_missed_frames: int = 20
    shape_library_path: Path = Path("data/shape_library.sqlite3")
    neural_shape_model_path: Path = Path("data/models/shape_net.pt")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
