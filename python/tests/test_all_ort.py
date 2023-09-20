# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr_onnxruntime import LoadImageError, RapidOCR

engine = RapidOCR()
tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"


def test_only_det():
    result, _ = engine(img_path, use_det=True, use_cls=False, use_rec=False)

    assert len(result) == 18
    assert result[0][0] == [5.0, 2.0]


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
    assert len(result) == 16


def test_cls_rec():
    img_path = tests_dir / "text_cls.jpg"
    result, _ = engine(img_path, use_det=False, use_cls=True, use_rec=True)

    assert len(result) == 1
    assert result[0][0] == "韩国小馆"


def test_det_cls_rec():
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result, _ = engine(img)
    assert result[0][1] == "正品促销"
    assert len(result) == 16


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
    result, _ = engine(str(img_path))
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_input_bytes():
    with open(img_path, "rb") as f:
        result, _ = engine(f.read())
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_input_path():
    result, _ = engine(img_path)
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_input_parameters():
    img_path = tests_dir / "ch_en_num.jpg"
    engine = RapidOCR(text_score=1)
    result, _ = engine(img_path)

    assert result is None


def test_input_det_parameters():
    with pytest.raises(FileNotFoundError) as exc_info:
        engine = RapidOCR(det_model_path="1.onnx")
        result, _ = engine(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_cls_parameters():
    with pytest.raises(FileNotFoundError) as exc_info:
        engine = RapidOCR(cls_model_path="1.onnx")
        result, _ = engine(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_rec_parameters():
    with pytest.raises(FileNotFoundError) as exc_info:
        engine = RapidOCR(rec_model_path="1.onnx")
        result, _ = engine(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_three_ndim_two_channel():
    img_npy = tests_dir / "two_dim_image.npy"
    image_array = np.load(str(img_npy))
    result, _ = engine(image_array)

    assert len(result) == 87


def test_input_three_ndim_one_channel():
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, 0]
    img = img[..., None]  # (H, W, 1)

    result, _ = engine(img)

    assert result[0][1] == "正品促销"
    assert len(result) == 16
