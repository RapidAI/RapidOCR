# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

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


def test_vis_only_det(engine):
    img_path = tests_dir / "ch_en_num.jpg"
    result = engine(img_path, use_det=True, use_cls=False, use_rec=False)
    vis_img = result.vis()
    assert vis_img.shape[2] == 3


def test_vis_only_rec(engine):
    img_path = tests_dir / "text_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)
    vis_img = result.vis()
    assert vis_img.shape[2] == 3


def test_vis_only_cls(engine):
    img_path = tests_dir / "text_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=False)
    vis_img = result.vis()
    assert vis_img.shape[2] == 3


def test_vis_det_cls_rec(engine):
    img_path = tests_dir / "ch_en_num.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=False)
    vis_img = result.vis()
    assert vis_img.shape[2] == 3
