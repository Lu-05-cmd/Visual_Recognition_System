from collections import Counter
from math import hypot

from visual_system.models import BoundingBox, Detection


class CentroidTracker:
    def __init__(self, max_distance: float = 120.0, max_missed_frames: int = 20) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self._next_id = 1
        self._centers: dict[int, tuple[float, float]] = {}
        self._velocities: dict[int, tuple[float, float]] = {}
        self._missed: dict[int, int] = {}
        self._colors: dict[int, str] = {}
        self._color_votes: dict[int, Counter[str]] = {}
        self._snapshots: dict[int, tuple[BoundingBox, float, str]] = {}
        self._shapes: dict[int, tuple[str, str]] = {}
        self._shape_votes: dict[int, Counter[str]] = {}
        self._solid_shape_votes: dict[int, Counter[str]] = {}
        self._shape_pending: dict[int, dict[str, Counter[str]]] = {}
        self._shape_locked: dict[int, tuple[str, str]] = {}
        self._box_centers: dict[int, tuple[float, float]] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched = set(self._centers)
        for detection in detections:
            center = detection.bounding_box.center
            compatible = [
                track_id
                for track_id in unmatched
                if detection.color == self._colors[track_id]
                or detection.color == "unknown"
                or self._colors[track_id] == "unknown"
            ]
            candidate_pool = compatible or list(unmatched)
            candidate = min(
                candidate_pool,
                key=lambda track_id: hypot(
                    center[0] - self._predicted_center(track_id)[0],
                    center[1] - self._predicted_center(track_id)[1],
                ),
                default=None,
            )
            candidate_distance = (
                hypot(
                    center[0] - self._predicted_center(candidate)[0],
                    center[1] - self._predicted_center(candidate)[1],
                )
                if candidate is not None
                else float("inf")
            )
            allowed_distance = (
                self.max_distance
                * (
                    1
                    + min(
                        self._missed[candidate] / max(self.max_missed_frames, 1),
                        0.5,
                    )
                )
                if candidate is not None
                else 0
            )
            color_matches = (
                candidate is not None
                and (
                    detection.color == self._colors[candidate]
                    or detection.color == "unknown"
                    or self._colors[candidate] == "unknown"
                )
            )
            # A short color-recognition glitch must not create a new serial.
            # Only allow this fallback when the center is still close.
            color_fallback = (
                candidate is not None
                and not color_matches
                and candidate_distance <= self.max_distance * 0.5
            )
            if (
                candidate is not None
                and (color_matches or color_fallback)
                and candidate_distance <= allowed_distance
            ):
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
                    self._color_votes[candidate][detection.color] += 1
                locked_color, locked_votes = self._color_votes[candidate].most_common(1)[0]
                if locked_votes >= 2:
                    self._colors[candidate] = locked_color
                detection.color = self._colors[candidate]
                detection.shape = self._confirm_shape(
                    candidate, "outline", detection.shape, self._shape_votes[candidate]
                )
                detection.solid_shape = self._confirm_shape(
                    candidate,
                    "solid",
                    detection.solid_shape,
                    self._solid_shape_votes[candidate],
                )
                self._shapes[candidate] = (detection.shape, detection.solid_shape)
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
                self._color_votes[self._next_id] = Counter(
                    {detection.color: 1}
                )
                self._snapshots[self._next_id] = (
                    detection.bounding_box,
                    detection.confidence,
                    detection.label,
                )
                self._shapes[self._next_id] = ("unknown", "unknown")
                self._shape_votes[self._next_id] = Counter(
                    {detection.shape: 0} if detection.shape != "unknown" else {}
                )
                self._solid_shape_votes[self._next_id] = Counter(
                    {detection.solid_shape: 0}
                    if detection.solid_shape != "unknown"
                    else {}
                )
                self._shape_pending[self._next_id] = {
                    "outline": Counter({detection.shape: 1})
                    if detection.shape != "unknown"
                    else Counter(),
                    "solid": Counter({detection.solid_shape: 1})
                    if detection.solid_shape != "unknown"
                    else Counter(),
                }
                self._shape_locked[self._next_id] = ("unknown", "unknown")
                if detection.shape != "unknown":
                    detection.shape = "unknown"
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
                del self._color_votes[track_id]
                del self._snapshots[track_id]
                del self._shapes[track_id]
                del self._shape_votes[track_id]
                del self._solid_shape_votes[track_id]
                del self._shape_pending[track_id]
                del self._shape_locked[track_id]
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
        track_id: int,
        kind: str,
        observed: str,
        votes: Counter[str],
        required_observations: int = 3,
    ) -> str:
        """Require consecutive evidence to confirm or change a track's shape."""
        index = 0 if kind == "outline" else 1
        committed = self._shape_locked[track_id][index]
        pending = self._shape_pending[track_id][kind]
        if observed == "unknown":
            return committed
        if observed == committed:
            pending.clear()
            return committed

        pending[observed] += 1
        consecutive = pending[observed]
        for label in tuple(pending):
            if label != observed:
                del pending[label]
        if consecutive < required_observations:
            return committed

        votes[observed] += consecutive
        pending.clear()
        locked = list(self._shape_locked[track_id])
        locked[index] = observed
        self._shape_locked[track_id] = (locked[0], locked[1])
        return observed

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
