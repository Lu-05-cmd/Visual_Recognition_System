import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from visual_system.services.shape_recognizer import ShapeRecognizer

SHAPE_LABELS = frozenset(
    {
        "circle",
        "oval",
        "triangle",
        "rectangle",
        "square",
        "pentagon",
        "hexagon",
        "polygon",
        "sphere",
        "cylinder",
        "cone",
        "box",
    }
)
SOLID_SHAPE_LABELS = frozenset({"sphere", "cylinder", "cone", "box"})
LEGACY_SOLID_SHAPE_LABELS = frozenset({"cube", "cuboid"})
SHAPE_IMAGE_LABELS = SHAPE_LABELS | LEGACY_SOLID_SHAPE_LABELS


class ShapeFeatureLibrary:
    """Persistent few-shot library of labeled object-shape feature points."""

    axes = ("compactness", "aspect_symmetry", "contour_complexity")
    # Legacy three-feature threshold; new samples use the richer descriptor.
    distance_threshold = 0.16
    descriptor_distance_threshold = 0.32
    descriptor_version = 2

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.image_root = self.path.parent / "shape_library"
        for label in SHAPE_LABELS:
            (self.image_root / label).mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS shape_samples (
                    id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    image_hash TEXT NOT NULL,
                    image_path TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(label, image_hash)
                )
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(shape_samples)")
            }
            if "image_path" not in columns:
                connection.execute("ALTER TABLE shape_samples ADD COLUMN image_path TEXT")
            if "descriptor" not in columns:
                connection.execute("ALTER TABLE shape_samples ADD COLUMN descriptor TEXT")
            if "descriptor_version" not in columns:
                connection.execute(
                    "ALTER TABLE shape_samples ADD COLUMN "
                    "descriptor_version INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_shape_samples_label ON shape_samples(label)"
            )
            # Upgrade existing labeled images into the richer contour model.
            legacy_samples = connection.execute(
                "SELECT id, image_path FROM shape_samples "
                "WHERE descriptor IS NULL OR descriptor_version < ?",
                (self.descriptor_version,),
            ).fetchall()
            for sample_id, stored_path in legacy_samples:
                if not stored_path:
                    continue
                sample_path = self.path.parent / stored_path
                if not sample_path.is_file():
                    continue
                image = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
                extracted = ShapeRecognizer._extract_contour(image) if image is not None else None
                descriptor = self._descriptor(extracted) if extracted else None
                if descriptor is not None:
                    connection.execute(
                        "UPDATE shape_samples SET descriptor = ?, "
                        "descriptor_version = ? WHERE id = ?",
                        (json.dumps(descriptor), self.descriptor_version, sample_id),
                    )

    def add_image(
        self,
        image: np.ndarray,
        label: str,
    ) -> tuple[bool, bool, tuple[float, float, float]]:
        if label in {"cube", "cuboid"}:
            label = "box"
        if label not in SHAPE_LABELS:
            raise ValueError("Hình dạng không nằm trong danh sách hỗ trợ")
        extracted = ShapeRecognizer._extract_contour(image)
        if extracted is None:
            raise ValueError("Không tách được đường viền vật thể từ ảnh")
        coordinates = self.coordinates(extracted[1])
        if coordinates is None:
            raise ValueError("Đường viền ảnh không đủ thông tin để tạo đặc trưng")
        descriptor = self._descriptor(extracted)
        image_bytes = np.asarray(image.shape, dtype=np.int64).tobytes()
        image_bytes += np.ascontiguousarray(image).tobytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        encoded_image = cv2.imencode(".png", image)[1].tobytes()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT id, label, image_path FROM shape_samples "
                "WHERE image_hash = ? LIMIT 1",
                (image_hash,),
            ).fetchone()
            if existing:
                sample_id, existing_label, stored_path = existing
                existing_image = self.path.parent / stored_path if stored_path else None
                restored = existing_image is None or not existing_image.is_file()
                if restored:
                    image_path = self._store_image(
                        existing_label,
                        image_hash,
                        encoded_image,
                    )
                    connection.execute(
                        "UPDATE shape_samples SET image_path = ? WHERE id = ?",
                        (image_path, sample_id),
                    )
                return False, restored, coordinates

            image_path = self._store_image(label, image_hash, encoded_image)
            try:
                connection.execute(
                    """
                    INSERT INTO shape_samples(
                        label, x, y, z, image_hash, image_path, descriptor,
                        descriptor_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        label,
                        *coordinates,
                        image_hash,
                        image_path,
                        json.dumps(descriptor),
                        self.descriptor_version,
                    ),
                )
            except Exception:
                (self.path.parent / image_path).unlink(missing_ok=True)
                raise
            return True, False, coordinates

    def classify(self, extracted, labels: set[str] | None = None) -> str | None:
        if extracted is None:
            return None
        query = self._descriptor(extracted)
        legacy_query = self.coordinates(extracted[1])
        if query is None or legacy_query is None:
            return None

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT label, x, y, z, descriptor, descriptor_version FROM shape_samples"
            ).fetchall()
        if not rows:
            return None

        distances = []
        for label, x, y, z, descriptor_json, version in rows:
            label = {"cube": "box", "cuboid": "box"}.get(label, label)
            if labels is not None and label not in labels:
                continue
            if descriptor_json and version == self.descriptor_version:
                try:
                    reference = np.asarray(json.loads(descriptor_json), dtype=np.float32)
                    distance = self._descriptor_distance(query, reference)
                    threshold = self.descriptor_distance_threshold
                except (ValueError, TypeError):
                    continue
            else:
                distance = float(np.linalg.norm(np.subtract(legacy_query, (x, y, z))))
                threshold = self.distance_threshold
            distances.append((distance, label, threshold))
        if not distances:
            return None
        distances.sort(key=lambda item: item[0])
        nearest_distance, nearest_label, threshold = distances[0]
        if nearest_distance > threshold:
            return None
        # A small, distance-weighted k-nearest-neighbor vote is less sensitive
        # to one mislabeled or atypical sample than a single nearest point.
        class_distances: dict[str, list[tuple[float, float]]] = {}
        for distance, label, allowed in distances:
            class_distances.setdefault(label, []).append((distance, allowed))
        class_scores = sorted(
            (
                float(np.mean([item[0] for item in sorted(samples)[:3]])),
                label,
                min(item[1] for item in samples),
            )
            for label, samples in class_distances.items()
        )
        best_score, best_label, best_threshold = class_scores[0]
        if best_score > best_threshold:
            return None
        if len(class_scores) > 1 and class_scores[1][0] - best_score < 0.025:
            return None
        return best_label

    def classify_relaxed(
        self,
        extracted,
        labels: set[str] | None = None,
        relaxed_threshold: float = 0.4,
    ) -> str | None:
        """Fallback classifier with relaxed thresholds for low-confidence detections."""
        if extracted is None:
            return None
        query = self._descriptor(extracted)
        legacy_query = self.coordinates(extracted[1])
        if query is None or legacy_query is None:
            return None

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT label, x, y, z, descriptor, descriptor_version FROM shape_samples"
            ).fetchall()
        if not rows:
            return None

        distances = []
        for label, x, y, z, descriptor_json, version in rows:
            label = {"cube": "box", "cuboid": "box"}.get(label, label)
            if labels is not None and label not in labels:
                continue
            if descriptor_json and version == self.descriptor_version:
                try:
                    reference = np.asarray(json.loads(descriptor_json), dtype=np.float32)
                    distance = self._descriptor_distance(query, reference)
                except (ValueError, TypeError):
                    continue
            else:
                distance = float(np.linalg.norm(np.subtract(legacy_query, (x, y, z))))
            distances.append((distance, label))
        if not distances:
            return None
        distances.sort(key=lambda item: item[0])
        nearest_distance, nearest_label = distances[0]
        if nearest_distance > relaxed_threshold:
            return None
        return nearest_label

    @classmethod
    def _descriptor(cls, extracted) -> tuple[float, ...] | None:
        if extracted is None:
            return None
        _, contour = extracted
        base = cls.coordinates(contour)
        if base is None:
            return None
        _, _, width, height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        rotated_width, rotated_height = cv2.minAreaRect(contour)[1]
        extent = area / max(rotated_width * rotated_height, 1)
        solidity = area / max(hull_area, 1)
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        hu_features = [
            float(np.clip(-np.sign(value) * np.log10(max(abs(float(value)), 1e-30)), -12, 12) / 12)
            for value in hu[:4]
        ]
        hull = cv2.convexHull(contour)
        vertices = len(cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True))
        vertex_feature = min(vertices, 16) / 16
        roundness = ShapeRecognizer._ellipse_roundness(contour)
        roundness = float(np.clip(roundness, 0, 1))
        return tuple(
            float(value)
            for value in (*base, extent, solidity, vertex_feature, roundness, *hu_features[:3])
        )

    @staticmethod
    def _descriptor_distance(first, second) -> float:
        if len(first) != len(second):
            return float("inf")
        weights = np.asarray((1.0, 1.0, 0.7, 1.0, 0.8, 0.7, 0.8, 0.1, 0.1, 0.08))
        difference = np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32)
        return float(np.sqrt(np.sum(weights[: len(difference)] * difference**2)))

    @staticmethod
    def coordinates(contour: np.ndarray) -> tuple[float, float, float] | None:
        perimeter = float(cv2.arcLength(contour, True))
        area = float(cv2.contourArea(contour))
        if perimeter <= 0 or area <= 0:
            return None

        _, _, width, height = cv2.boundingRect(contour)
        compactness = float(np.clip(4 * np.pi * area / (perimeter * perimeter), 0, 1))
        rotated_width, rotated_height = cv2.minAreaRect(contour)[1]
        aspect_symmetry = min(rotated_width, rotated_height) / max(rotated_width, rotated_height, 1)
        hull = cv2.convexHull(contour)
        approximation = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True)
        contour_complexity = min(len(approximation), 12) / 12
        return (compactness, aspect_symmetry, contour_complexity)

    def summary(self) -> dict:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, label, x, y, z, created_at, image_path
                FROM shape_samples
                ORDER BY label, id
                """
            ).fetchall()
        classes = Counter()
        for row in rows:
            classes[{"cube": "box", "cuboid": "box"}.get(row[1], row[1])] += 1
        groups: dict[str, list[dict]] = {}
        for row in rows:
            label = {"cube": "box", "cuboid": "box"}.get(row[1], row[1])
            sample = {
                "id": row[0],
                "label": label,
                "coordinates": [row[2], row[3], row[4]],
                "created_at": row[5],
                "image_url": None,
            }
            if row[6]:
                image_path = self.path.parent / row[6]
                if image_path.is_file():
                    sample["image_url"] = (
                        f"/api/shapes/images/{row[1]}/{image_path.name}"
                    )
            groups.setdefault(label, []).append(sample)
        return {
            "axes": list(self.axes),
            "sample_count": len(rows),
            "classes": dict(classes),
            "groups": groups,
            "samples": [sample for samples in groups.values() for sample in samples],
        }

    def delete_sample(self, sample_id: int) -> str | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT label, image_path FROM shape_samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
            if row is None:
                return None
            label, stored_path = row
            connection.execute("DELETE FROM shape_samples WHERE id = ?", (sample_id,))

        folder = (self.image_root / label).resolve()
        if stored_path:
            image_path = (self.path.parent / stored_path).resolve()
            if image_path.parent == folder:
                image_path.unlink(missing_ok=True)
        try:
            folder.rmdir()
        except OSError:
            pass
        return label

    def _store_image(self, label: str, image_hash: str, encoded_image: bytes) -> str:
        image_path = self.image_root / label / f"{image_hash}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(encoded_image)
        return image_path.relative_to(self.path.parent).as_posix()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
