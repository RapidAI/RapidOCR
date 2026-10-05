# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
from pytest import mark

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"


@mark.parametrize(
    "engine_type",
    [
        EngineType.ONNXRUNTIME,
        EngineType.PADDLE,
        EngineType.OPENVINO,
        EngineType.MNN,
    ],
)
def test_ppocrv5_cls_mobile(engine_type):
    engine = RapidOCR(
        params={
            "Cls.ocr_version": OCRVersion.PPOCRV5,
            "Cls.model_type": ModelType.MOBILE,
            "Cls.engine_type": engine_type,
        }
    )
    img_path = tests_dir / "text_cls.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=False)

    assert result.cls_res is not None
    assert result.cls_res[0][0] == "180"


def test_ppocrv5_cls_server():
    engine = RapidOCR(
        params={
            "Cls.ocr_version": OCRVersion.PPOCRV5,
            "Cls.model_type": ModelType.SERVER,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
        }
    )
    img_path = tests_dir / "text_cls.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=False)

    assert result.cls_res is not None
    assert result.cls_res[0][0] == "180"


def test_ppocrv4_cls_long_line():
    # A long line used to be squeezed into the 192x48 input of the v2.0 model,
    # which then missed that it was upside down (#744).
    engine = RapidOCR()
    img = cv2.imread(str(tests_dir / "test_letterbox_like.jpg"))
    thresh = engine.cfg.Cls.cls_thresh

    result = engine(img, use_det=False, use_cls=True, use_rec=False)
    label, score = result.cls_res[0]
    assert not ("180" in label and score > thresh)

    rotated = cv2.rotate(img, cv2.ROTATE_180)
    result = engine(rotated, use_det=False, use_cls=True, use_rec=False)
    label, score = result.cls_res[0]
    assert "180" in label and score > thresh
