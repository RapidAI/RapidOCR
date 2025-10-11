# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import RapidOCR

tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"


@pytest.fixture()
def engine():
    engine = RapidOCR()
    return engine


def test_only_det(engine):
    result = engine(img_path, use_det=True, use_cls=False, use_rec=False)
    assert len(result) == 18


def test_only_cls(engine):
    img_path = tests_dir / "text_cls.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=False)
    assert len(result) == 1
    assert result.cls_res[0][0] == "180"


def test_only_rec(engine):
    img_path = tests_dir / "text_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)
    assert len(result) == 1
    assert result.txts[0] == "韩国小馆"


def test_det_rec(engine):
    result = engine(img_path, use_det=True, use_cls=False, use_rec=True)
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_cls_rec(engine):
    img_path = tests_dir / "text_cls.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=True)

    assert result is not None
    assert len(result) == 1
    assert result.txts[0] == "韩国小馆"


def test_det_cls_rec(engine):
    img = cv2.imread(str(img_path))

    result = engine(img)
    assert result is not None
    assert len(result) == 18
    assert result.txts[0] == "正品促销"
