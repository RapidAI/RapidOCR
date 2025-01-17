# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import logging
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr_torch import LoadImageError, RapidOCR

from .base_module import download_file

engine = RapidOCR()
tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"
package_name = "rapidocr_torch"


def test_long_img():
    img_url = "https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/long.jpeg"
    img_path = tests_dir / "long.jpeg"
    download_file(img_url, save_path=img_path)
    result, _ = engine(img_path)

    assert result is not None
    assert len(result) >= 53

    img_path.unlink()



def test_mode_one_img():
    img_path = tests_dir / "issue_170.png"
    result, _ = engine(img_path)
    assert result[0][1] == "TEST"


@pytest.mark.parametrize(
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
def test_transparent_img(img_name: str, gt: str):
    img_path = tests_dir / img_name
    result, _ = engine(str(img_path))
    assert result[0][1] == gt


@pytest.mark.parametrize(
    "img_name,gt_len,gt_first_len",
    [
        ("test_without_det.jpg", 1, "在中国作家协会第三届儿童文学"),
    ],
)
def test_letterbox_like(img_name, gt_len, gt_first_len):
    img_path = tests_dir / img_name
    result, _ = engine(str(img_path))

    assert len(result) == gt_len
    assert result[0][1].lower() == gt_first_len.lower()


def test_only_det():
    result, _ = engine(str(img_path), use_det=True, use_cls=False, use_rec=False)

    assert len(result) == 18


def test_only_cls():
    img_path = tests_dir / "text_cls.jpg"
    result, _ = engine(img_path, use_det=False, use_cls=True, use_rec=False)
    assert len(result) == 1
    assert result[0][0] == "0"


def test_only_rec():
    img_path = tests_dir / "text_rec.jpg"
    result, _ = engine(img_path, use_det=False, use_cls=False, use_rec=True)
    assert len(result) == 1
    assert result[0][0] == "韩国小馆"


def test_det_rec():
    result, _ = engine(img_path, use_det=True, use_cls=False, use_rec=True)
    assert result[0][1] == "正品促销"
    assert len(result) == 18


def test_cls_rec():
    img_path = tests_dir / "text_cls.jpg"
    result, _ = engine(img_path, use_det=False, use_cls=True, use_rec=True)

    assert result is not None
    assert len(result) == 1
    assert result[0][0] == "韩国小馆"


def test_det_cls_rec():
    img = cv2.imread(img_path)

    result, _ = engine(img)
    assert result is not None
    assert result[0][1] == "正品促销"
    assert len(result) == 18


def test_empty():
    img = None
    with pytest.raises(LoadImageError) as exc_info:
        engine(img)
        raise LoadImageError
    assert exc_info.type is LoadImageError


def test_zeros():
    img = np.zeros([640, 640, 3], np.uint8)
    result, _ = engine(img)
    assert result is None


def test_input_str():
    result, _ = engine(img_path)
    assert result[0][1] == "正品促销"
    assert len(result) == 18


def test_input_bytes():
    with open(img_path, "rb") as f:
        result, _ = engine(f.read())
    assert result[0][1] == "正品促销"
    assert len(result) == 18


def test_input_path():
    result, _ = engine(img_path)
    assert result[0][1] == "正品促销"
    assert len(result) == 18


def test_input_parameters():
    img_path = tests_dir / "ch_en_num.jpg"
    engine = RapidOCR(text_score=1)
    result, _ = engine(img_path)

    assert result is None


def test_input_det_parameters():
    with pytest.raises(FileNotFoundError) as exc_info:
        engine = RapidOCR(det_model_path="1.pth")
        result, _ = engine(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_cls_parameters():
    with pytest.raises(FileNotFoundError) as exc_info:
        engine = RapidOCR(cls_model_path="1.pth")
        result, _ = engine(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_rec_parameters():
    with pytest.raises(FileNotFoundError) as exc_info:
        engine = RapidOCR(rec_model_path="1.pth")
        result, _ = engine(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_three_ndim_two_channel():
    img_npy = tests_dir / "two_dim_image.npy"
    image_array = np.load(str(img_npy))
    result, _ = engine(image_array)

    assert result is not None
    assert len(result) == 1
    assert result[0][1] == "TREND PLOT REPORT"


def test_input_three_ndim_one_channel():
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, 0]
    img = img[..., None]  # (H, W, 1)

    result, _ = engine(img)

    assert len(result) >= 17


@pytest.mark.parametrize(
    "img_name,words",
    [
        (
            "black_font_color_transparent.png",
            ["我", "是", "中", "国", "人"],
        ),
        (
            "issue_170.png",
            ["T", "E", "S", "T"],
        ),
    ],
)
def test_word_ocr(img_name: str, words: List[str]):
    img_path = tests_dir / img_name
    result, _ = engine(str(img_path), return_word_box=True)
    assert result[0][4] == words
