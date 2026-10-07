# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from pytest import mark

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
from rapidocr.cal_rec_boxes import CalRecBoxes
from rapidocr.ch_ppocr_rec import TextRecOutput
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode

test_dir = root_dir / "tests" / "test_files"


@pytest.fixture()
def engine():
    engine = RapidOCR(
        params={
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Det.model_type": ModelType.MOBILE,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Cls.ocr_version": OCRVersion.PPOCRV4,
            "Cls.model_type": ModelType.MOBILE,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.CH,
        }
    )
    return engine


def test_txts_equal_words(engine):
    img_path = test_dir / "check_return_word_len.jpeg"
    result = engine(img_path, return_word_box=True)

    assert len(result.txts) == len(result.word_results)


@mark.parametrize(
    "img_name,words",
    [
        (
            "black_font_color_transparent.png",
            ("我", "是", "中", "国", "人"),
        ),
        (
            "text_vertical_words.png",
            ("已", "取", "之", "時", "不", "參", "一", "人", "見", "而"),
        ),
    ],
)
def test_cn_word_ocr(engine, img_name: str, words: List[str]):
    img_path = test_dir / img_name
    result = engine(img_path, return_word_box=True)
    txts, _, _ = list(zip(*result.word_results[0]))
    assert txts == words


@mark.parametrize(
    "img_name,words",
    [("issue_170.png", "TEST"), ("return_word_debug.jpg", "3F1")],
)
def test_en_word_ocr(engine, img_name: str, words: str):
    img_path = test_dir / img_name
    result = engine(img_path, return_word_box=True)
    txts, _, _ = list(zip(*result.word_results[0]))
    assert txts[0] == words


def test_en_return_single_char_box(engine):
    img_path = test_dir / "en.jpg"
    result = engine(img_path, return_word_box=True, return_single_char_box=True)
    txts, _, _ = list(zip(*result.word_results[0]))
    assert txts[0] == "3"


@mark.parametrize(
    "chars,return_single_char_box,expected",
    [
        ("ab cd", False, [("ab", 0.75), ("cd", 0.65)]),
        ("ab cd", True, [("a", 0.9), ("b", 0.6), ("c", 0.8), ("d", 0.5)]),
        ("中 文", False, [("中", 0.9), ("文", 0.6)]),
    ],
)
def test_word_conf_from_own_chars(
    chars: str, return_single_char_box: bool, expected: List[Tuple[str, float]]
):
    decoder = CTCLabelDecode(character=["a", "b", "c", "d", "中", "文"])

    # One character every other step, blanks between. Spaces score 0.99.
    probs = iter([0.9, 0.6, 0.8, 0.5])
    steps = [(decoder.dict[c], 0.99 if c == " " else next(probs)) for c in chars]
    preds = np.zeros((1, 2 * len(steps), len(decoder.character)), dtype=np.float32)
    preds[0, :, 0] = 0.95
    for i, (char_idx, prob) in enumerate(steps):
        preds[0, 2 * i, 0] = 0.0
        preds[0, 2 * i, char_idx] = prob

    line_results, word_infos = decoder(preds, return_word_box=True)
    assert line_results[0][0] == chars

    img = np.zeros((48, 320, 3), dtype=np.uint8)
    box = np.array([[0, 0], [320, 0], [320, 48], [0, 48]], dtype=np.float32)
    rec_res = TextRecOutput(imgs=[img], txts=(chars,), word_results=tuple(word_infos))
    rec_res = CalRecBoxes()([img], [box], rec_res, return_single_char_box)

    words = [(txt, conf) for txt, conf, _ in rec_res.word_results[0]]
    assert words == expected
