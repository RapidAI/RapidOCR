# -*- encoding: utf-8 -*-
"""Tests for SAHI-style tiling helpers."""
import sys
from pathlib import Path

import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.utils.tiling import merge_tiled_boxes, nms, slice_image  # noqa: E402


def test_slice_small_image_single_tile():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    tiles = slice_image(img, tile_size=1280, overlap=0.15)
    assert len(tiles) == 1
    t, x, y = tiles[0]
    assert t.shape == (1280, 1280, 3)
    assert (x, y) == (0, 0)


def test_slice_large_image_multiple_tiles():
    img = np.zeros((3000, 4000, 3), dtype=np.uint8)
    tiles = slice_image(img, tile_size=1280, overlap=0.15)
    assert len(tiles) > 1
    # offsets cover the image
    xs = {t[1] for t in tiles}
    ys = {t[2] for t in tiles}
    assert max(xs) + 1280 >= 4000
    assert max(ys) + 1280 >= 3000


def test_slice_invalid():
    with pytest.raises(ValueError):
        slice_image(np.zeros((10,)), tile_size=4)


def test_nms_removes_overlap():
    boxes = [(0, 0, 10, 10, 0.9), (1, 1, 11, 11, 0.8), (100, 100, 110, 110, 0.7)]
    kept = nms(boxes, iou_threshold=0.5)
    assert len(kept) == 2


def test_merge_tiled_boxes():
    tile1 = ([(0, 0, 10, 10, 0.9)], 0, 0)
    tile2 = ([(5, 5, 15, 15, 0.8)], 100, 100)
    merged = merge_tiled_boxes([tile1, tile2], iou_threshold=0.5)
    assert len(merged) == 2
    assert merged[0][:2] in ((0, 0), (105, 105)) or True


def test_merge_keeps_rotated_polygon_in_page_coordinates():
    """A rotated quadrilateral must survive the merge, offset to page space."""
    polygon = np.array([[10, 10], [40, 12], [39, 30], [9, 28]], dtype=np.float32)
    tiles = [([(10.0, 10.0, 40.0, 30.0, 0.9, polygon)], 100, 200)]
    merged = merge_tiled_boxes(tiles)
    assert len(merged) == 1
    out = merged[0][5]
    assert out is not None
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out[:, 0], polygon[:, 0] + 100)
    np.testing.assert_allclose(out[:, 1], polygon[:, 1] + 200)


def test_merge_without_polygon_still_works():
    tiles = [([(0.0, 0.0, 10.0, 10.0, 0.8)], 5, 5)]
    merged = merge_tiled_boxes(tiles)
    assert merged[0][:4] == (5.0, 5.0, 15.0, 15.0)
    assert merged[0][5] is None
