# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from pytest import mark

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import LoadImageError, RapidOCR

test_dir = root_dir / "tests" / "test_files"
img_path = test_dir / "ch_en_num.jpg"


@pytest.fixture()
def engine():
    engine = RapidOCR()
    return engine


def test_exif_transpose(engine):
    img_path = test_dir / "img_exif_orientation.jpg"
    result = engine(img_path, use_cls=False)
    assert result.txts[0] == "我是中国人"


@mark.parametrize(
    "img_name,gt",
    [
        (
            "black_font_color_transparent.png",
            "我是中国人",
        ),
        (
            "white_font_color_transparent.png",
            "我是中国人",
        ),
    ],
)
def test_transparent_img(engine, img_name: str, gt: str):
    img_path = test_dir / img_name
    result = engine(img_path)
    assert result.txts[0] == gt


def test_long_img(engine):
    img_url = "https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/long.jpeg"
    result = engine(img_url)

    assert result is not None
    assert len(result.boxes) >= 53


def test_full_black_img(engine):
    img_path = test_dir / "empty_black.jpg"
    result = engine(img_path)
    assert result.img is None
    assert result.boxes is None


def test_img_url_input(engine):
    img_url = "https://github.com/RapidAI/RapidOCR/blob/a9bb7c1f44b6e00556ada90ac588f020d7637c4b/python/tests/test_files/ch_en_num.jpg?raw=true"
    result = engine(img_url)
    assert result.txts is not None
    assert result.txts[0] == "正品促销"


def test_empty(engine):
    img = None
    with pytest.raises(LoadImageError) as exc_info:
        engine(img)
        raise LoadImageError
    assert exc_info.type is LoadImageError


def test_zeros(engine):
    img = np.zeros([640, 640, 3], np.uint8)
    result = engine(img)
    assert result.boxes is None


def test_input_str(engine):
    result = engine(str(img_path))
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_input_bytes(engine):
    with open(img_path, "rb") as f:
        result = engine(f.read())
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_input_path(engine):
    result = engine(img_path)
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_input_parameters(engine):
    result = engine(img_path, text_score=1.0)
    assert result.boxes is None


def test_input_three_ndim_two_channel(engine):
    img_npy = test_dir / "two_dim_image.npy"
    image_array = np.load(str(img_npy))
    result = engine(image_array)

    assert result is not None
    assert len(result) == 1
    assert result.txts[0] == "TREND PLOT REPORT"


def test_input_three_ndim_one_channel(engine):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, 0]
    img = img[..., None]  # (H, W, 1)

    result = engine(img)
    assert len(result) >= 17


def test_mode_one_img(engine):
    img_path = test_dir / "issue_170.png"
    result = engine(img_path)
    assert result.txts[0] == "TEST"


@mark.parametrize(
    "img_name,gt_len,gt_first_len",
    [
        (
            "test_letterbox_like.jpg",
            2,
            "A：：取决于所使用的执行提供者，它可能没有完全支持模型中的所有操作。回落到CPU操作可能会导致性能速度的下降。此外，即使一个操作是由CUDAeXecution",
        ),
        ("test_without_det.jpg", 1, "在中国作家协会第三届儿童文学"),
    ],
)
def test_letterbox_like(engine, img_name, gt_len, gt_first_len):
    img_path = test_dir / img_name
    result = engine(img_path)

    assert len(result) == gt_len
    assert result.txts[0].lower() == gt_first_len.lower()
