from dataclasses import dataclass, field
from enum import StrEnum


class InspectionStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class Detection:
    label: str
    confidence: float
    bounding_box: BoundingBox
    color: str = "unknown"
    shape: str = "unknown"
    track_id: int | None = None
    serial_number: int | None = None
    inspection: InspectionStatus = InspectionStatus.UNKNOWN
    defects: list[str] = field(default_factory=list)
    solid_shape: str = "unknown"


@dataclass
class FrameResult:
    frame_id: int
    width: int
    height: int
    detections: list[Detection]
