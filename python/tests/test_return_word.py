# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path
from typing import List

import pytest
from pytest import mark

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import RapidOCR

tests_dir = root_dir / "tests" / "test_files"


@pytest.fixture()
def engine():
    engine = RapidOCR()
    return engine


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
    img_path = tests_dir / img_name
    result = engine(img_path, return_word_box=True)
    txts, _, _ = list(zip(*result.word_results[0]))
    assert txts == words


@mark.parametrize(
    "img_name,words",
    [("issue_170.png", "TEST"), ("return_word_debug.jpg", "3F1")],
)
def test_en_word_ocr(engine, img_name: str, words: str):
    img_path = tests_dir / img_name
    result = engine(img_path, return_word_box=True)
    txts, _, _ = list(zip(*result.word_results[0]))
    assert txts[0] == words


def test_en_return_single_char_box(engine):
    img_path = tests_dir / "en.jpg"
    result = engine(img_path, return_word_box=True, return_single_char_box=True)
    txts, _, _ = list(zip(*result.word_results[0]))
    assert txts[0] == "3"
