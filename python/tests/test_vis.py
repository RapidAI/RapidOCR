# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageOps

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
from rapidocr.utils import vis_res

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
    result = engine(img_path, use_det=True, use_cls=True, use_rec=True)
    vis_img = result.vis()
    assert vis_img.shape[2] == 3


ARABIC_TEXT = "امروز هوا خوبه"


def ink_box(img):
    return ImageOps.invert(img.convert("L")).getbbox()


def test_vis_draws_rtl_text_with_khatt(monkeypatch):
    khatt = pytest.importorskip("khatt")
    drawn = []
    draw_text = khatt.draw_text

    def spy(img, xy, txt, **kwargs):
        drawn.append(txt)
        return draw_text(img, xy, txt, **kwargs)

    monkeypatch.setattr(vis_res.khatt, "draw_text", spy)
    engine = RapidOCR(
        params={
            "Rec.lang_type": LangRec.ARABIC,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        }
    )
    result = engine(tests_dir / "arabic.png", use_det=False, use_cls=False, use_rec=True)
    vis_img = result.vis()
    assert vis_img.shape[2] == 3
    assert drawn == list(result.txts)


def test_rtl_line_starts_at_the_right_edge_and_fits_its_box():
    pytest.importorskip("khatt")
    vis = vis_res.VisRes(lang_type=LangRec.ARABIC)
    img = Image.new("RGB", (300, 60), (255, 255, 255))
    box = [[20, 10], [280, 10], [280, 50], [20, 50]]
    vis.draw_line(img, ImageDraw.Draw(img), box, ARABIC_TEXT, 40)

    left, top, right, bottom = ink_box(img)
    assert right == 280
    assert left >= 20 and top >= 10 and bottom <= 50


def test_ltr_line_is_drawn_by_pillow(monkeypatch):
    khatt = pytest.importorskip("khatt")
    monkeypatch.setattr(vis_res.khatt, "draw_text", lambda *args, **kwargs: pytest.fail("khatt used"))
    vis = vis_res.VisRes()
    img = Image.new("RGB", (300, 60), (255, 255, 255))
    vis.draw_line(img, ImageDraw.Draw(img), [[20, 10], [280, 10], [280, 50], [20, 50]], "Hello", 32)
    assert ink_box(img) is not None


def test_rtl_line_without_khatt_is_drawn_by_pillow(monkeypatch):
    monkeypatch.setattr(vis_res, "khatt", None)
    vis = vis_res.VisRes(lang_type=LangRec.ARABIC)
    img = Image.new("RGB", (300, 60), (255, 255, 255))
    vis.draw_line(img, ImageDraw.Draw(img), [[20, 10], [280, 10], [280, 50], [20, 50]], ARABIC_TEXT, 32)
    assert ink_box(img)[0] >= 20
