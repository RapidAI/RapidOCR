# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest


root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr_onnxruntime import RapidOCR

rapid_ocr = RapidOCR()
tests_dir = root_dir / 'test_files'


def test_normal():
    image_path = tests_dir / 'ch_en_num.jpg'
    img = cv2.imread(str(image_path))
    result, _ = rapid_ocr(img)
    assert result[0][1] == '正品促销'
    assert len(result) == 17


def test_empty():
    img = None
    with pytest.raises(AttributeError) as exc_info:
        rapid_ocr(img)
        raise AttributeError
    assert exc_info.type is AttributeError


def test_zeros():
    img = np.zeros([640, 640, 3])
    result, _ = rapid_ocr(img)
    assert result is None
