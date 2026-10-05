from collections import Counter, deque
from math import hypot

from visual_system.models import BoundingBox, Detection

SHAPE_CONFIRMATION_FRAMES = 5
COLOR_HISTORY_FRAMES = 9
COLOR_CONFIRMATION_VOTES = 5


class CentroidTracker:
    def __init__(self, max_distance: float = 120.0, max_missed_frames: int = 20) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self._next_id = 1
        self._centers: dict[int, tuple[float, float]] = {}
        self._velocities: dict[int, tuple[float, float]] = {}
        self._missed: dict[int, int] = {}
        self._colors: dict[int, str] = {}
        self._color_history: dict[int, deque[str]] = {}
        self._snapshots: dict[int, tuple[BoundingBox, float, str]] = {}
        self._solid_shape_pending: dict[int, deque[str]] = {}
        self._solid_shape_locked: dict[int, str] = {}
        self._box_centers: dict[int, tuple[float, float]] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched = set(self._centers)
        candidate_pairs = []
        for detection_index, detection in enumerate(detections):
            center = detection.bounding_box.center
            for track_id in unmatched:
                predicted_center = self._predicted_center(track_id)
                distance = hypot(
                    center[0] - predicted_center[0],
                    center[1] - predicted_center[1],
                )
                allowed_distance = self.max_distance * (
                    1
                    + min(
                        self._missed[track_id] / max(self.max_missed_frames, 1),
                        0.5,
                    )
                )
                color_matches = (
                    detection.color == self._colors[track_id]
                    or detection.color == "unknown"
                    or self._colors[track_id] == "unknown"
                )
                if distance <= allowed_distance:
                    color_penalty = 0 if color_matches else self.max_distance * 0.1
                    candidate_pairs.append(
                        (distance + color_penalty, detection_index, track_id)
                    )

        matches: dict[int, int] = {}
        matched_tracks: set[int] = set()
        for _, detection_index, track_id in sorted(candidate_pairs):
            if detection_index in matches or track_id in matched_tracks:
                continue
            matches[detection_index] = track_id
            matched_tracks.add(track_id)

        for detection_index, detection in enumerate(detections):
            center = detection.bounding_box.center
            candidate = matches.get(detection_index)
            if candidate is not None:
                detection.track_id = candidate
                snapshot_box = self._snapshots[candidate][0]
                stable_box = self._update_box_size(snapshot_box, detection.bounding_box)
                self._snapshots[candidate] = (
                    stable_box,
                    detection.confidence,
                    detection.label,
                )
                smoothed_center = self._smooth_box_center(candidate, center)
                detection.bounding_box = self._move_box_with_fixed_size(
                    smoothed_center, stable_box.width, stable_box.height
                )
                # Recognition and box geometry must come from this frame; a
                # previous track snapshot may belong to a different object.
                if detection.color != "unknown":
                    history = self._color_history[candidate]
                    history.append(detection.color)
                    color_votes = Counter(history)
                    stable_color, stable_votes = color_votes.most_common(1)[0]
                    if stable_votes >= COLOR_CONFIRMATION_VOTES and (
                        stable_color != self._colors[candidate]
                        or color_votes[stable_color] > color_votes[self._colors[candidate]]
                    ):
                        self._colors[candidate] = stable_color
                detection.color = self._colors[candidate]
                detection.solid_shape = self._confirm_shape(
                    detection.solid_shape,
                    self._solid_shape_pending[candidate],
                    self._solid_shape_locked[candidate],
                )
                self._solid_shape_locked[candidate] = detection.solid_shape
                previous_center = self._centers[candidate]
                self._centers[candidate] = center
                previous_velocity = self._velocities[candidate]
                displacement = (
                    center[0] - previous_center[0],
                    center[1] - previous_center[1],
                )
                self._velocities[candidate] = (
                    previous_velocity[0] * 0.5 + displacement[0] * 0.5,
                    previous_velocity[1] * 0.5 + displacement[1] * 0.5,
                )
                self._missed[candidate] = 0
                unmatched.remove(candidate)
            else:
                detection.track_id = self._next_id
                self._centers[self._next_id] = center
                self._box_centers[self._next_id] = center
                self._velocities[self._next_id] = (0.0, 0.0)
                self._missed[self._next_id] = 0
                self._colors[self._next_id] = detection.color
                self._color_history[self._next_id] = deque(
                    [detection.color] if detection.color != "unknown" else [],
                    maxlen=COLOR_HISTORY_FRAMES,
                )
                self._snapshots[self._next_id] = (
                    detection.bounding_box,
                    detection.confidence,
                    detection.label,
                )
                self._solid_shape_pending[self._next_id] = deque(
                    [detection.solid_shape]
                    if detection.solid_shape != "unknown"
                    else [],
                    maxlen=SHAPE_CONFIRMATION_FRAMES,
                )
                self._solid_shape_locked[self._next_id] = "unknown"
                if detection.solid_shape != "unknown":
                    detection.solid_shape = "unknown"
                self._next_id += 1
        for track_id in unmatched:
            self._missed[track_id] += 1
            if self._missed[track_id] > self.max_missed_frames:
                del self._centers[track_id]
                del self._box_centers[track_id]
                del self._velocities[track_id]
                del self._missed[track_id]
                del self._colors[track_id]
                del self._color_history[track_id]
                del self._snapshots[track_id]
                del self._solid_shape_pending[track_id]
                del self._solid_shape_locked[track_id]
        return detections

    def _predicted_center(self, track_id: int) -> tuple[float, float]:
        center = self._centers[track_id]
        velocity = self._velocities[track_id]
        elapsed_frames = self._missed[track_id] + 1
        return (
            center[0] + velocity[0] * elapsed_frames,
            center[1] + velocity[1] * elapsed_frames,
        )

    def _smooth_box_center(
        self, track_id: int, measured: tuple[float, float], alpha: float = 0.65
    ) -> tuple[float, float]:
        """Low-pass filter detection jitter while following real object motion."""
        previous = self._box_centers.get(track_id, measured)
        smoothed = (
            previous[0] * (1 - alpha) + measured[0] * alpha,
            previous[1] * (1 - alpha) + measured[1] * alpha,
        )
        self._box_centers[track_id] = smoothed
        return smoothed

    def _confirm_shape(
        self,
        observed: str,
        pending: deque[str],
        committed: str,
        required_observations: int = SHAPE_CONFIRMATION_FRAMES,
    ) -> str:
        """Lock a solid label after a majority of the last five valid observations."""
        if committed != "unknown":
            pending.clear()
            return committed
        if observed == "unknown":
            return committed
        if observed == committed:
            pending.clear()
            return committed

        pending.append(observed)
        votes = Counter(pending)
        label, count = votes.most_common(1)[0]
        if count < (required_observations // 2) + 1:
            return committed

        pending.clear()
        return label

    @staticmethod
    def _move_box_with_fixed_size(
        center: tuple[float, float],
        width: int,
        height: int,
    ) -> BoundingBox:
        return BoundingBox(
            x=round(center[0] - width / 2),
            y=round(center[1] - height / 2),
            width=width,
            height=height,
        )

    @staticmethod
    def _update_box_size(previous: BoundingBox, current: BoundingBox) -> BoundingBox:
        """Grow a track box when later frames reveal more of the same object."""
        return BoundingBox(
            x=current.x,
            y=current.y,
            width=max(previous.width, current.width),
            height=max(previous.height, current.height),
        )
