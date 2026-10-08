# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import ctypes
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from pytest import mark

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import (
    EngineType,
    LangRec,
    LoadImageError,
    ModelType,
    OCRVersion,
    RapidOCR,
)
from rapidocr.utils.load_image import LoadImage, MemoryImage, RawMemoryImage
from rapidocr.utils.parse_parameters import ParseParams

test_dir = root_dir / "tests" / "test_files"
img_path = test_dir / "ch_en_num.jpg"
config_path = root_dir / "rapidocr" / "config.yaml"


@pytest.fixture()
def engine():
    engine = RapidOCR(
        params={
            "Det.ocr_version": OCRVersion.PPOCRV4,
            "Det.model_type": ModelType.MOBILE,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Cls.ocr_version": OCRVersion.PPOCRV4,
            "Cls.model_type": ModelType.MOBILE,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.CH,
        }
    )
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
    img_url = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/resources/test_files/long.jpeg"
    result = engine(img_url)

    assert result is not None
    assert len(result.boxes) >= 53


def test_full_black_img(engine):
    img_path = test_dir / "empty_black.jpg"
    result = engine(img_path)
    assert result.img is None
    assert result.boxes is None


def test_img_url_input(engine):
    img_url = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/resources/test_files/ch_en_num.jpg"
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


def test_input_memory_image(engine):
    """An encoded image can be passed by address and byte length."""
    encoded = img_path.read_bytes()
    buffer = ctypes.create_string_buffer(encoded)
    memory_img = MemoryImage(ctypes.addressof(buffer), len(encoded))

    result = engine(memory_img)

    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_input_memory_image_matches_bytes(engine):
    encoded = img_path.read_bytes()
    buffer = ctypes.create_string_buffer(encoded)
    memory_img = MemoryImage(ctypes.addressof(buffer), len(encoded))

    expected = engine(encoded)
    actual = engine(memory_img)

    assert actual.txts == expected.txts
    assert actual.scores == expected.scores
    np.testing.assert_allclose(actual.boxes, expected.boxes)


@pytest.mark.parametrize(
    "address,length",
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_input_memory_image_invalid_address_or_length(address, length):
    loader = LoadImage()

    with pytest.raises(LoadImageError, match="MemoryImage loading failed"):
        loader(MemoryImage(address, length))


def test_input_memory_image_invalid_encoded_data():
    encoded = b"not an image"
    buffer = ctypes.create_string_buffer(encoded)
    loader = LoadImage()

    with pytest.raises(LoadImageError, match="MemoryImage loading failed"):
        loader(MemoryImage(ctypes.addressof(buffer), len(encoded)))


def make_raw_memory_image(img, pixel_format, stride=None, bottom_up=False):
    height, width = img.shape[:2]
    row_bytes = img.shape[1] * (1 if img.ndim == 2 else img.shape[2])
    stride = row_bytes if stride is None else stride
    rows = img[::-1] if bottom_up else img
    raw = bytearray(stride * height)
    for row_index, row in enumerate(rows):
        row_data = row.tobytes()
        raw[row_index * stride : row_index * stride + row_bytes] = row_data
    buffer = ctypes.create_string_buffer(bytes(raw))
    memory_img = RawMemoryImage(
        ctypes.addressof(buffer),
        len(raw),
        width,
        height,
        pixel_format=pixel_format,
        stride=stride,
        bottom_up=bottom_up,
    )
    return memory_img, buffer


def test_input_raw_memory_image(engine):
    img = cv2.imread(str(img_path))
    memory_img, buffer = make_raw_memory_image(img, "BGR")

    result = engine(memory_img)

    assert buffer is not None  # Keep the backing memory alive through inference.
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


@pytest.mark.parametrize(
    "pixel_format,source",
    [
        ("BGR", np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)),
        ("RGB", np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8)),
        ("BGRA", np.array([[[1, 2, 3, 0], [4, 5, 6, 255]]], dtype=np.uint8)),
        ("BGRX", np.array([[[1, 2, 3, 8], [4, 5, 6, 9]]], dtype=np.uint8)),
        ("RGBA", np.array([[[3, 2, 1, 0], [6, 5, 4, 255]]], dtype=np.uint8)),
        ("RGBX", np.array([[[3, 2, 1, 8], [6, 5, 4, 9]]], dtype=np.uint8)),
    ],
)
def test_raw_memory_image_converts_to_bgr(pixel_format, source):
    memory_img, buffer = make_raw_memory_image(source, pixel_format)

    result = LoadImage()(memory_img)

    assert buffer is not None
    np.testing.assert_array_equal(
        result, np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    )


def test_raw_memory_image_supports_gray_stride_and_bottom_up():
    gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    memory_img, buffer = make_raw_memory_image(
        gray, "GRAY8", stride=4, bottom_up=True
    )

    result = LoadImage()(memory_img)

    assert buffer is not None
    expected = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"address": 0}, "invalid memory address"),
        ({"width": 0}, "width and height must be positive"),
        ({"pixel_format": "YUV"}, "unsupported pixel format"),
        ({"stride": 5}, "stride .* is smaller than row size"),
        ({"length": 8}, "buffer is too small"),
    ],
)
def test_raw_memory_image_rejects_invalid_metadata(kwargs, error):
    raw = bytes(12)
    buffer = ctypes.create_string_buffer(raw)
    params = {
        "address": ctypes.addressof(buffer),
        "length": len(raw),
        "width": 2,
        "height": 2,
        "pixel_format": "BGR",
        "stride": 6,
    }
    params.update(kwargs)

    with pytest.raises(LoadImageError, match=error):
        LoadImage()(RawMemoryImage(**params))


def test_input_path(engine):
    result = engine(img_path)
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_input_parameters(engine):
    result = engine(img_path, text_score=1.0)
    assert result.boxes is None


@mark.parametrize("params", [{"text_score": 0.5}, {"Global": 0.5}, {"Unknown.key": 1}])
def test_invalid_init_parameters(params):
    cfg = ParseParams.load(config_path)

    with pytest.raises(ValueError):
        ParseParams.update_batch(cfg, params)


def test_nested_init_parameters():
    cfg = ParseParams.load(config_path)
    ParseParams.update_batch(cfg, {"EngineConfig.onnxruntime.use_cuda": True})

    assert cfg.EngineConfig.onnxruntime.use_cuda is True


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
