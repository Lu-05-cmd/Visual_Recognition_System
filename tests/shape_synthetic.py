from __future__ import annotations

import cv2
import numpy as np


FLAT_SHAPES = ("circle", "oval", "triangle", "square", "rectangle", "pentagon", "hexagon")
SOLID_SHAPES = ("sphere", "cylinder", "cone", "cube")


def _polygon(name: str, center: tuple[int, int], radius: int) -> np.ndarray:
    x, y = center
    if name in {"triangle", "cone"}:
        points = [(x, y - radius), (x + radius, y + radius), (x - radius, y + radius)]
    elif name == "square":
        points = [(x-radius, y-radius), (x+radius, y-radius), (x+radius, y+radius), (x-radius, y+radius)]
    elif name == "rectangle":
        points = [(x-radius, y-radius//2), (x+radius, y-radius//2), (x+radius, y+radius//2), (x-radius, y+radius//2)]
    else:
        count = 5 if name == "pentagon" else 6
        points = [
            (round(x + radius * np.cos(2 * np.pi * i / count - np.pi / 2)),
             round(y + radius * np.sin(2 * np.pi * i / count - np.pi / 2)))
            for i in range(count)
        ]
    return np.asarray(points, dtype=np.int32)


def make_scene(
    name: str,
    seed: int = 0,
    width: int = 320,
    height: int = 240,
) -> np.ndarray:
    """Generate a labeled BGR scene with randomized scale, position, light and background."""
    rng = np.random.default_rng(seed)
    backgrounds = ((238, 238, 238), (218, 225, 224), (32, 35, 38), (90, 92, 96))
    background = np.asarray(backgrounds[seed % len(backgrounds)], dtype=np.float32)
    scene = np.empty((height, width, 3), dtype=np.float32)
    gradient = np.linspace(-8, 8, width, dtype=np.float32)[None, :, None]
    scene[:] = background + gradient
    scene = np.clip(scene, 0, 255).astype(np.uint8)

    cx = int(width * (0.35 + 0.3 * ((seed * 7) % 11) / 10))
    cy = int(height * (0.35 + 0.3 * ((seed * 5) % 11) / 10))
    radius = int(min(width, height) * (0.16 + 0.04 * (seed % 4)))
    base = np.asarray((140, 82, 40), dtype=np.float32)
    object_mask = np.zeros((height, width), dtype=np.uint8)
    object_layer = np.zeros_like(scene)

    if name in FLAT_SHAPES:
        angle = (seed % 8) * 11.25
        if name in {"circle", "oval"}:
            axes = (radius, radius) if name == "circle" else (radius, max(12, radius // 2))
            cv2.ellipse(object_mask, (cx, cy), axes, angle, 0, 360, 255, -1, cv2.LINE_AA)
        else:
            points = _polygon(name, (cx, cy), radius).astype(np.float32)
            rotation = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            points = cv2.transform(points[None, :, :], rotation)[0].astype(np.int32)
            cv2.fillPoly(object_mask, [points], 255, cv2.LINE_AA)
        object_layer[:] = base
    elif name == "sphere":
        cv2.circle(object_mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
        yy, xx = np.mgrid[:height, :width]
        radial = np.sqrt((xx-cx)**2 + (yy-cy)**2) / max(radius, 1)
        light = np.clip(1.2 - 0.55 * radial + 0.12 * (cx-xx) / max(radius, 1), 0.45, 1.35)
        object_layer = np.clip(base[None, None, :] * light[:, :, None], 0, 255).astype(np.uint8)
    elif name == "cylinder":
        body_w, body_h = int(radius * 1.35), int(radius * 1.55)
        cv2.rectangle(object_mask, (cx-body_w//2, cy-body_h//2), (cx+body_w//2, cy+body_h//2), 255, -1)
        cv2.ellipse(object_mask, (cx, cy-body_h//2), (body_w//2, max(8, radius//4)), 0, 0, 360, 255, -1)
        cv2.ellipse(object_mask, (cx, cy+body_h//2), (body_w//2, max(8, radius//4)), 0, 0, 360, 255, -1)
        object_layer[:] = base
        cv2.ellipse(object_layer, (cx, cy-body_h//2), (body_w//2, max(8, radius//4)), 0, 0, 360, (168, 106, 60), -1)
        cv2.ellipse(
            object_layer,
            (cx, cy-body_h//2),
            (body_w//2, max(8, radius//4)),
            0,
            0,
            360,
            (88, 54, 29),
            max(2, radius // 22),
        )
    elif name == "cone":
        points = _polygon(name, (cx, cy), radius)
        cv2.fillPoly(object_mask, [points], 255, cv2.LINE_AA)
        object_layer[:] = 0
        for row in range(max(0, cy-radius), min(height, cy+radius+1)):
            columns = np.flatnonzero(object_mask[row] > 0)
            if not len(columns):
                continue
            center_column = (columns[0] + columns[-1]) / 2
            half_width = max((columns[-1] - columns[0]) / 2, 1)
            edge_distance = np.abs(columns - center_column) / half_width
            light = np.clip(1.18 - 0.55 * edge_distance, 0.5, 1.2)
            object_layer[row, columns] = np.clip(base[None, :] * light[:, None], 0, 255)
    elif name == "cube":
        top = np.asarray([(cx, cy-radius), (cx+radius, cy-radius//2), (cx, cy), (cx-radius, cy-radius//2)], np.int32)
        left = np.asarray([(cx-radius, cy-radius//2), (cx, cy), (cx, cy+radius), (cx-radius, cy+radius//2)], np.int32)
        right = np.asarray([(cx, cy), (cx+radius, cy-radius//2), (cx+radius, cy+radius//2), (cx, cy+radius)], np.int32)
        silhouette = np.asarray([(cx, cy-radius), (cx+radius, cy-radius//2), (cx+radius, cy+radius//2), (cx, cy+radius), (cx-radius, cy+radius//2), (cx-radius, cy-radius//2)], np.int32)
        cv2.fillPoly(object_mask, [silhouette], 255, cv2.LINE_AA)
        cv2.fillPoly(object_layer, [top], (170, 118, 72), cv2.LINE_AA)
        cv2.fillPoly(object_layer, [left], (140, 82, 40), cv2.LINE_AA)
        cv2.fillPoly(object_layer, [right], (82, 49, 26), cv2.LINE_AA)

    alpha = object_mask.astype(np.float32)[:, :, None] / 255
    scene = (scene * (1-alpha) + object_layer * alpha).astype(np.uint8)
    brightness = 0.82 + (seed % 5) * 0.08
    scene = np.clip(scene.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    noise = rng.normal(0, 1.5 + seed % 3, scene.shape).astype(np.float32)
    return np.clip(scene.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def scene_box(name: str, seed: int, width: int = 320, height: int = 240) -> tuple[int, int, int, int]:
    cx = int(width * (0.35 + 0.3 * ((seed * 7) % 11) / 10))
    cy = int(height * (0.35 + 0.3 * ((seed * 5) % 11) / 10))
    radius = int(min(width, height) * (0.16 + 0.04 * (seed % 4)))
    half_width = round(radius * 1.3)
    half_height = round(radius * 1.3)
    return (
        max(0, cx - half_width),
        max(0, cy - half_height),
        min(width, cx + half_width),
        min(height, cy + half_height),
    )
