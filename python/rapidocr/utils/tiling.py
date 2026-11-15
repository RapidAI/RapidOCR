"""Sliced inference (SAHI-style) helpers for large images (blueprints, big tables).

Implements the tiling strategy from SPANISH_MULTIDOC.md section 4.9 and
OPTIMIZATION_STRATEGY.md section 5.3: partition a high-resolution image into
overlapping tiles, run detection per tile, then merge boxes in global coords via
NMS.

This module is backend-agnostic: it operates on numpy arrays and box lists so it
can wrap any detector (RapidOCR Det, ONNX, etc.).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def slice_image(
    img: np.ndarray,
    tile_size: int = 1280,
    overlap: float = 0.15,
) -> List[Tuple[np.ndarray, int, int]]:
    if img.ndim < 2:
        raise ValueError("img must be at least 2D")
    h, w = img.shape[0], img.shape[1]
    step = max(1, int(tile_size * (1 - overlap)))
    tiles: List[Tuple[np.ndarray, int, int]] = []
    ys = list(range(0, max(1, h - tile_size + 1), step))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))
    ys = sorted(set(ys))
    xs = sorted(set(xs))
    for y in ys:
        for x in xs:
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tile = img[y:y2, x:x2]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                pad = np.zeros((tile_size, tile_size) + img.shape[2:], dtype=img.dtype)
                pad[: tile.shape[0], : tile.shape[1]] = tile
                tile = pad
            tiles.append((tile, x, y))
    return tiles


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: List[Tuple], iou_threshold: float = 0.5) -> List[Tuple]:
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][4] if len(boxes[i]) > 4 else 0, reverse=True)
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j in suppressed or j == i:
                continue
            if _iou(boxes[i][:4], boxes[j][:4]) >= iou_threshold:
                suppressed.add(j)
    return [boxes[i] for i in keep]


def merge_tiled_boxes(
    tiled_results: List[Tuple[List[Tuple], int, int]],
    iou_threshold: float = 0.5,
) -> List[Tuple]:
    """Merge per-tile detections into page coordinates.

    Each detection is `(x1, y1, x2, y2, score[, polygon])`. The polygon is the
    rotated quadrilateral the detector produced; it travels with the detection so
    the caller can crop the original shape instead of its axis-aligned envelope.
    """
    global_boxes = []
    for boxes, off_x, off_y in tiled_results:
        for b in boxes:
            x1, y1, x2, y2 = b[:4]
            score = b[4] if len(b) > 4 else 1.0
            polygon = b[5] if len(b) > 5 else None
            if polygon is not None:
                polygon = np.asarray(polygon, dtype=np.float32) + np.array(
                    [off_x, off_y], dtype=np.float32
                )
            global_boxes.append(
                (x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y, score, polygon)
            )
    return nms(global_boxes, iou_threshold)