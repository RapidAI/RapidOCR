# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest


root_dir = Path(__file__).resolve().parent.parent
print(root_dir)
sys.path.append(str(root_dir))

from rapidocr_onnxruntime import RapidOCR, LoadImageError

rapid_ocr = RapidOCR()
tests_dir = root_dir / 'tests' / 'test_files'


def test_normal():
    image_path = tests_dir / 'ch_en_num.jpg'
    img = cv2.imread(str(image_path))
    result, _ = rapid_ocr(img)
    assert result[0][1] == '正品促销'
    assert len(result) == 17


def test_empty():
    img = None
    with pytest.raises(LoadImageError) as exc_info:
        rapid_ocr(img)
        raise LoadImageError
    assert exc_info.type is LoadImageError


def test_zeros():
    img = np.zeros([640, 640, 3])
    result, _ = rapid_ocr(img)
    assert result is None


def test_input_str():
    image_path = tests_dir / 'ch_en_num.jpg'
    result, _ = rapid_ocr(str(image_path))
    assert result[0][1] == '正品促销'
    assert len(result) == 17


def test_input_bytes():
    image_path = tests_dir / 'ch_en_num.jpg'
    with open(image_path, 'rb') as f:
        result, _ = rapid_ocr(f.read())
    assert result[0][1] == '正品促销'
    assert len(result) == 17


def test_input_path():
    image_path = tests_dir / 'ch_en_num.jpg'
    result, _ = rapid_ocr(image_path)
    assert result[0][1] == '正品促销'
    assert len(result) == 17


def test_input_parameters():
    image_path = tests_dir / 'ch_en_num.jpg'
    rapid_ocr = RapidOCR(text_score=1)
    result, _ = rapid_ocr(image_path)

    assert result is None


def test_input_det_parameters():
    image_path = tests_dir / 'ch_en_num.jpg'
    with pytest.raises(FileNotFoundError) as exc_info:
        rapid_ocr = RapidOCR(det_model_path='1.onnx')
        result, _ = rapid_ocr(image_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_cls_parameters():
    image_path = tests_dir / 'ch_en_num.jpg'
    with pytest.raises(FileNotFoundError) as exc_info:
        rapid_ocr = RapidOCR(cls_model_path='1.onnx')
        result, _ = rapid_ocr(image_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_rec_parameters():
    image_path = tests_dir / 'ch_en_num.jpg'
    with pytest.raises(FileNotFoundError) as exc_info:
        rapid_ocr = RapidOCR(rec_model_path='1.onnx')
        result, _ = rapid_ocr(image_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError
