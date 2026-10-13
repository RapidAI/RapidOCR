# -*- encoding: utf-8 -*-
# @Author: Mohammad Hijjawi
from typing import List, Optional, Tuple

import numpy as np
import pytest
from rapidocr import RapidOCR
from rapidocr.ch_ppocr_cls import TextClsOutput
from rapidocr.ch_ppocr_det import TextDetOutput
from rapidocr.ch_ppocr_rec import TextRecOutput
from rapidocr.ch_ppocr_rec.typings import WordInfo, WordType

LINE_TXT = "我是人"

# A 200x20 horizontal line and a 20x200 vertical line in a 400x400 image.
HORIZONTAL_BOX = [[100, 50], [300, 50], [300, 70], [100, 70]]
VERTICAL_BOX = [[100, 100], [120, 100], [120, 300], [100, 300]]


def make_word_info() -> WordInfo:
    return WordInfo(
        words=[list(LINE_TXT)],
        word_cols=[[2, 15, 28]],
        word_types=[WordType.CN],
        line_txt_len=30,
        confs=[0.9, 0.9, 0.9],
    )


def run_word_boxes(
    boxes: List[List[List[int]]],
    txts: Tuple[str, ...],
    cls_res: Optional[List[Tuple[str, float]]],
):
    engine = RapidOCR(params={"Global.return_word_box": True})

    # The crop handed to recognition is always laid out horizontally.
    crops = [np.zeros((20, 200, 3), dtype=np.uint8) for _ in boxes]
    det_res = TextDetOutput(
        boxes=np.array(boxes, dtype=np.float32), scores=[0.9] * len(boxes)
    )
    cls_out = TextClsOutput(img_list=crops, cls_res=cls_res)
    rec_res = TextRecOutput(
        imgs=crops,
        txts=txts,
        scores=tuple(0.9 for _ in txts),
        word_results=tuple(make_word_info() for _ in txts),
    )
    return engine.build_final_output(
        np.zeros((400, 400, 3), dtype=np.uint8),
        det_res,
        cls_out,
        rec_res,
        crops,
        {"preprocess": {"ratio_h": 1.0, "ratio_w": 1.0}},
    )


def word_centers(word_line) -> List[Tuple[float, float]]:
    return [tuple(np.array(points).mean(axis=0)) for _, _, points in word_line]


@pytest.mark.parametrize(
    "box,cls_res,axis,reading_direction",
    [
        (HORIZONTAL_BOX, None, 0, 1),
        (HORIZONTAL_BOX, [("0", 0.99)], 0, 1),
        # below cls_thresh the crop is not rotated, so neither are the boxes
        (HORIZONTAL_BOX, [("180", 0.5)], 0, 1),
        # issue #328: upside-down text is read right to left in the image
        (HORIZONTAL_BOX, [("180", 0.99)], 0, -1),
        (VERTICAL_BOX, [("0", 0.99)], 1, 1),
        (VERTICAL_BOX, [("180", 0.99)], 1, -1),
    ],
)
def test_word_boxes_follow_cls_rotation(box, cls_res, axis, reading_direction):
    result = run_word_boxes([box], (LINE_TXT,), cls_res)

    word_line = result.word_results[0]
    assert "".join(txt for txt, _, _ in word_line) == LINE_TXT

    centers = [center[axis] for center in word_centers(word_line)]
    assert np.all(reading_direction * np.diff(centers) > 0)


def test_rotated_word_boxes_are_mirrored_within_line():
    upright = run_word_boxes([HORIZONTAL_BOX], (LINE_TXT,), [("0", 0.99)])
    flipped = run_word_boxes([HORIZONTAL_BOX], (LINE_TXT,), [("180", 0.99)])

    upright_xs = [
        sorted({p[0] for p in points}) for _, _, points in upright.word_results[0]
    ]
    flipped_xs = [
        sorted({p[0] for p in points}) for _, _, points in flipped.word_results[0]
    ]
    # Each character's box is mirrored about the centre of the line.
    for (x0, x1), (fx0, fx1) in zip(upright_xs, flipped_xs):
        assert (fx0, fx1) == (400 - x1, 400 - x0)


def test_rotation_flags_stay_aligned_after_empty_lines_are_dropped():
    result = run_word_boxes(
        [HORIZONTAL_BOX, VERTICAL_BOX],
        ("", LINE_TXT),
        [("0", 0.99), ("180", 0.99)],
    )

    assert result.txts == (LINE_TXT,)
    ys = [center[1] for center in word_centers(result.word_results[0])]
    assert np.all(np.diff(ys) < 0)
