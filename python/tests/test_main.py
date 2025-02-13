# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import LoadImageError, RapidOCR
from rapidocr.inference_engine.base import InferSession

tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"


@pytest.fixture()
def engine():
    engine = RapidOCR()
    return engine


def get_engine(params: Optional[Dict[str, Any]] = None):
    if params:
        engine = RapidOCR(params=params)
        return engine

    engine = RapidOCR()
    return engine


def test_engine_openvino():
    engine = get_engine(params={"Global.with_openvino": True})
    result = engine(img_path)
    assert result.txts[0] == "正品促销"


# def test_engine_paddle():
#     engine = RapidOCR(
#         params={
#             "Global.with_paddle": True,
#             "Det.model_path": "tests/test_files/ch_ppocr_server_v2.0_det_infer.onnx",
#             "Rec.model_path": "tests/test_files/ch_ppocr_server_v2.0_rec_infer.onnx",
#         }
#     )
#     result = engine(img_path)
#     assert result.txts[0] == "正品促销"


def test_long_img(engine):
    img_url = "https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/long.jpeg"
    img_path = tests_dir / "long.jpeg"
    InferSession.download_file(img_url, save_path=img_path)
    result = engine(img_path)

    assert result is not None
    assert len(result.boxes) >= 53

    img_path.unlink()


def test_ort_cuda_warning(caplog):
    engine = RapidOCR(
        params={"Global.with_onnx": True, "EngineConfig.onnxruntime.use_cuda": True}
    )
    caplog.set_level(logging.WARNING)
    assert caplog.records[0].levelname == "WARNING"
    assert "CUDAExecutionProvider" in caplog.records[0].message


def test_ort_dml_warning(caplog):
    engine = RapidOCR(
        params={"Global.with_onnx": True, "EngineConfig.onnxruntime.use_dml": True}
    )
    caplog.set_level(logging.WARNING)

    assert caplog.records[0].levelname == "WARNING"
    assert "DirectML" in caplog.records[0].message


def test_mode_one_img(engine):
    img_path = tests_dir / "issue_170.png"
    result = engine(img_path)
    assert result.txts[0] == "TEST"


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
    result = engine(img_path)
    assert result.txts[0] == gt


@pytest.mark.parametrize(
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
def test_letterbox_like(img_name, gt_len, gt_first_len):
    img_path = tests_dir / img_name
    result = engine(img_path)

    assert len(result) == gt_len
    assert result.txts[0].lower() == gt_first_len.lower()


def test_only_det():
    result = engine(img_path, use_det=True, use_cls=False, use_rec=False)
    assert len(result) == 18


def test_only_cls():
    img_path = tests_dir / "text_cls.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=False)
    assert len(result) == 1
    assert result.cls_res[0][0] == "0"


def test_only_rec():
    img_path = tests_dir / "text_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)
    assert len(result) == 1
    assert result.txts[0] == "韩国小馆"


def test_det_rec():
    result = engine(img_path, use_det=True, use_cls=False, use_rec=True)
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_cls_rec():
    img_path = tests_dir / "text_cls.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=True)

    assert result is not None
    assert len(result) == 1
    assert result.txts[0] == "韩国小馆"


def test_det_cls_rec():
    img = cv2.imread(str(img_path))

    result = engine(img)
    assert result is not None
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_empty():
    img = None
    with pytest.raises(LoadImageError) as exc_info:
        engine(img)
        raise LoadImageError
    assert exc_info.type is LoadImageError


def test_zeros():
    img = np.zeros([640, 640, 3], np.uint8)
    result = engine(img)
    assert result.boxes is None


def test_input_str():
    result = engine(str(img_path))
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_input_bytes():
    with open(img_path, "rb") as f:
        result = engine(f.read())
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_input_path():
    result = engine(img_path)
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_input_parameters():
    img_path = tests_dir / "ch_en_num.jpg"
    engine = RapidOCR(params={"Global.text_score": 1})
    result = engine(img_path)
    assert result.boxes is None


def test_input_three_ndim_two_channel():
    img_npy = tests_dir / "two_dim_image.npy"
    image_array = np.load(str(img_npy))
    result = engine(image_array)

    assert result is not None
    assert len(result) == 1
    assert result.txts[0] == "TREND PLOT REPORT"


def test_input_three_ndim_one_channel():
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, 0]
    img = img[..., None]  # (H, W, 1)

    result = engine(img)
    assert len(result) >= 17


@pytest.mark.parametrize(
    "img_name,words",
    [
        (
            "black_font_color_transparent.png",
            ("我", "是", "中", "国", "人"),
        ),
        (
            "text_vertical_words.png",
            (
                "已",
                "取",
                "之",
                "時",
                "不",
                "參",
                "一",
                "人",
                "見",
                "而",
                "是",
                "非",
                "不",
                "得",
                "問",
                "之",
                "人",
                "要",
                "取",
                "之",
                "有",
                "是",
                "是",
                "非",
                "非",
                "之",
                "士",
                "師",
                "也",
            ),
        ),
        (
            "issue_170.png",
            ("T", "E", "S", "T"),
        ),
    ],
)
def test_word_ocr(img_name: str, words: List[str]):
    img_path = tests_dir / img_name
    result = engine(img_path, return_word_box=True)
    txts, _, _ = list(zip(*result.word_results))
    assert txts == words
