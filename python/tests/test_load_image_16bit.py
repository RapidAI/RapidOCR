# -*- encoding: utf-8 -*-
import io

import cv2
import numpy as np
import pytest
from PIL import Image
from rapidocr.utils.load_image import LoadImage


def _gray_ramp_8bit() -> np.ndarray:
    ramp = np.append(np.arange(0, 256, 4), 255).astype(np.uint8)
    return np.tile(ramp, (8, 1))


def _png_bytes(arr: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", arr)
    assert ok
    return enc.tobytes()


def _tiff_bytes(arr: np.ndarray, mode: str) -> bytes:
    dtype = ">u2" if mode == "I;16B" else "<u2"
    img = Image.frombytes(mode, arr.shape[::-1], arr.astype(dtype).tobytes())
    buf = io.BytesIO()
    img.save(buf, "TIFF")
    return buf.getvalue()


@pytest.mark.parametrize(
    "kind", ["ndarray", "ndarray_big_endian", "pil", "png_bytes", "tiff_be_bytes"]
)
def test_16bit_gray_matches_8bit(kind):
    gray8 = _gray_ramp_8bit()
    gray16 = gray8.astype(np.uint16) * 257

    loader = LoadImage()
    expected = loader(gray8)

    if kind == "ndarray":
        got = loader(gray16)
    elif kind == "ndarray_big_endian":
        got = loader(gray16.astype(">u2"))
    elif kind == "pil":
        got = loader(Image.open(io.BytesIO(_png_bytes(gray16))))
    elif kind == "png_bytes":
        got = loader(_png_bytes(gray16))
    else:
        got = loader(_tiff_bytes(gray16, "I;16B"))

    assert got.dtype == np.uint8
    assert np.array_equal(got, expected)


def test_16bit_bgr_ndarray_keeps_channels():
    bgr8 = np.random.default_rng(0).integers(0, 256, (6, 9, 3), dtype=np.uint8)
    bgr8[0, 0] = (255, 0, 128)
    got = LoadImage()(bgr8.astype(np.uint16) * 257)
    assert got.dtype == np.uint8
    assert np.array_equal(got, bgr8)


def test_12bit_data_in_uint16_is_stretched_to_8bit():
    gray12 = np.tile(np.arange(0, 4096, 64, dtype=np.uint16), (8, 1))
    gray12[0, 0] = 4095
    got = LoadImage()(gray12)
    assert got.dtype == np.uint8
    assert got.min() == 0
    assert got.max() == 255


def test_uint16_with_8bit_values_is_unchanged():
    gray8 = _gray_ramp_8bit()
    loader = LoadImage()
    assert np.array_equal(loader(gray8.astype(np.uint16)), loader(gray8))
