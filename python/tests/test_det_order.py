# -*- encoding: utf-8 -*-
"""Scores must follow their boxes through the reading-order sort."""
import sys
from pathlib import Path

import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.ch_ppocr_det.main import TextDetector  # noqa: E402


def _box(x, y, w=10, h=10):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_sort_order_maps_boxes_and_scores_together():
    # Second box sits above the first: the sort must swap them, and the caller
    # applies the same permutation to the scores.
    boxes = np.array([_box(0, 50), _box(0, 0)], dtype=np.float32)
    scores = [0.9, 0.4]
    order = TextDetector.sort_order(boxes)
    assert list(order) == [1, 0]
    assert boxes[order][0][0][1] == 0
    assert [scores[i] for i in order] == [0.4, 0.9]


def test_sorted_boxes_still_orders_by_reading_order():
    boxes = np.array([_box(100, 0), _box(0, 0)], dtype=np.float32)
    ordered = TextDetector.sorted_boxes(boxes)
    assert ordered[0][0][0] == 0


def test_tiling_flags_exist_without_init():
    """The preprocessing tests build the detector with __new__."""
    detector = TextDetector.__new__(TextDetector)
    assert detector.use_tiling is False
    assert detector.tile_size == 1280
    assert detector.tile_min_side == 2000
