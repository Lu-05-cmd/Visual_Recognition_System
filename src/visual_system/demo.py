import cv2
import numpy as np


def create_demo_frame(frame_index: int = 0, width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (28, 32, 38)
    cv2.rectangle(
        frame,
        (100 + frame_index % 120, 170),
        (220 + frame_index % 120, 290),
        (30, 75, 220),
        -1,
    )
    cv2.circle(frame, (500 + frame_index % 90, 250), 65, (40, 190, 70), -1)
    cv2.rectangle(
        frame,
        (720 - frame_index % 100, 150),
        (850 - frame_index % 100, 300),
        (220, 150, 35),
        -1,
    )
    return frame
