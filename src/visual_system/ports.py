from typing import Protocol

import numpy as np

from visual_system.models import Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...
