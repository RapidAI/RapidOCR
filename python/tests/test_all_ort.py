# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
print(root_dir)
sys.path.append(str(root_dir))

from rapidocr_onnxruntime import LoadImageError, RapidOCR

rapid_ocr = RapidOCR()
tests_dir = root_dir / "tests" / "test_files"


def test_normal():
    img_path = tests_dir / "ch_en_num.jpg"
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result, _ = rapid_ocr(img)
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_empty():
    img = None
    with pytest.raises(LoadImageError) as exc_info:
        rapid_ocr(img)
        raise LoadImageError
    assert exc_info.type is LoadImageError


def test_zeros():
    img = np.zeros([640, 640, 3], np.uint8)
    result, _ = rapid_ocr(img)
    assert result is None


def test_input_str():
    img_path = tests_dir / "ch_en_num.jpg"
    result, _ = rapid_ocr(str(img_path))
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_input_bytes():
    img_path = tests_dir / "ch_en_num.jpg"
    with open(img_path, "rb") as f:
        result, _ = rapid_ocr(f.read())
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_input_path():
    img_path = tests_dir / "ch_en_num.jpg"
    result, _ = rapid_ocr(img_path)
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_input_parameters():
    img_path = tests_dir / "ch_en_num.jpg"
    rapid_ocr = RapidOCR(text_score=1)
    result, _ = rapid_ocr(img_path)

    assert result is None


def test_input_det_parameters():
    img_path = tests_dir / "ch_en_num.jpg"
    with pytest.raises(FileNotFoundError) as exc_info:
        rapid_ocr = RapidOCR(det_model_path="1.onnx")
        result, _ = rapid_ocr(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_cls_parameters():
    img_path = tests_dir / "ch_en_num.jpg"
    with pytest.raises(FileNotFoundError) as exc_info:
        rapid_ocr = RapidOCR(cls_model_path="1.onnx")
        result, _ = rapid_ocr(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_rec_parameters():
    img_path = tests_dir / "ch_en_num.jpg"
    with pytest.raises(FileNotFoundError) as exc_info:
        rapid_ocr = RapidOCR(rec_model_path="1.onnx")
        result, _ = rapid_ocr(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_three_ndim_two_channel():
    img_npy = tests_dir / "two_dim_image.npy"
    image_array = np.load(str(img_npy))
    result, _ = rapid_ocr(image_array)

    assert len(result) == 87


def test_input_three_ndim_one_channel():
    img_path = tests_dir / "ch_en_num.jpg"
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, 0]
    img = img[..., None]  # (H, W, 1)

    result, _ = rapid_ocr(img)

    assert result[0][1] == "正品促销"
    assert len(result) == 16
