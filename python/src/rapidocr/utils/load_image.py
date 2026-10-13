"""Image inputs accepted by rapidocr 3.x: path, URL, bytes, ndarray, PIL, memory, raw memory."""

from __future__ import annotations

import ctypes
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class LoadImageError(Exception):
    pass


class MemoryImage:
    """Encoded image passed by address and byte length, as on ``main``."""

    def __init__(self, address: int, length: int):
        self.address = address
        self.length = length


class RawMemoryImage:
    """Metadata needed to read an uncompressed pixel buffer by address, as on ``main``."""

    def __init__(
        self,
        address: int,
        length: int,
        width: int,
        height: int,
        pixel_format: str = "BGRA",
        stride: Optional[int] = None,
        bottom_up: bool = False,
    ):
        self.address = address
        self.length = length
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.stride = stride
        self.bottom_up = bottom_up


InputType = Union[
    str, np.ndarray, bytes, Path, Image.Image, MemoryImage, RawMemoryImage
]

_RAW_CHANNELS = {
    "GRAY": 1,
    "GRAY8": 1,
    "BGR": 3,
    "BGR24": 3,
    "RGB": 3,
    "RGB24": 3,
    "BGRA": 4,
    "BGRA32": 4,
    "BGRX": 4,
    "RGBA": 4,
    "RGBA32": 4,
    "RGBX": 4,
}


class LoadImage:
    def __call__(self, img: InputType) -> np.ndarray:
        if img is None or not isinstance(img, (str, Path, bytes, bytearray, np.ndarray, Image.Image, MemoryImage, RawMemoryImage)):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )
        # A raw buffer decodes to packed BGR, like an OpenCV ndarray.
        origin = np.ndarray if isinstance(img, RawMemoryImage) else type(img)
        array = self.load_img(img)
        return self.convert_img(array, origin)

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            text = str(img)
            if text.startswith("http://") or text.startswith("https://"):
                with urlopen(text, timeout=60) as response:
                    data = response.read()
                return self._decode_bytes(data)
            path = Path(text)
            if not path.exists():
                raise LoadImageError(f"{path} does not exist.")
            return self._decode_bytes(path.read_bytes())
        if isinstance(img, (bytes, bytearray)):
            return self._decode_bytes(bytes(img))
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            image = img
            if image.mode == "1":
                image = image.convert("L")
            image = ImageOps.exif_transpose(image) or image
            return np.asarray(image)
        if isinstance(img, MemoryImage):
            try:
                return self._load_memory(img)
            except LoadImageError:
                raise
            except Exception as exc:
                raise LoadImageError("MemoryImage loading failed.") from exc
        if isinstance(img, RawMemoryImage):
            try:
                return self.load_raw_memory_image(img)
            except Exception as exc:
                raise LoadImageError(f"RawMemoryImage loading failed: {exc}") from exc
        raise LoadImageError(f"{type(img)} is not supported!")

    @staticmethod
    def load_raw_memory_image(img: RawMemoryImage) -> np.ndarray:
        """Read a raw pixel buffer and return packed BGR (alpha/padding dropped)."""

        if img.address <= 0 or img.length <= 0:
            raise LoadImageError("invalid memory address or length")
        if img.width <= 0 or img.height <= 0:
            raise LoadImageError("width and height must be positive")

        pixel_format = str(img.pixel_format).upper()
        channels = _RAW_CHANNELS.get(pixel_format)
        if channels is None:
            supported = ", ".join(_RAW_CHANNELS)
            raise LoadImageError(
                f"unsupported pixel format {img.pixel_format!r}; supported: {supported}"
            )

        row_bytes = img.width * channels
        stride = row_bytes if img.stride is None else img.stride
        if stride < row_bytes:
            raise LoadImageError(f"stride ({stride}) is smaller than row size ({row_bytes})")

        required_length = stride * img.height
        if img.length < required_length:
            raise LoadImageError(
                f"buffer is too small: need {required_length} bytes, got {img.length}"
            )

        raw = ctypes.string_at(img.address, required_length)
        rows = np.frombuffer(raw, dtype=np.uint8).reshape(img.height, stride)
        pixels = rows[:, :row_bytes]
        if img.bottom_up:
            pixels = pixels[::-1]
        if channels == 1:
            gray = pixels.reshape(img.height, img.width)
            return np.ascontiguousarray(np.stack([gray, gray, gray], axis=-1))
        pixels = pixels.reshape(img.height, img.width, channels)[:, :, :3]
        if pixel_format.startswith("RGB"):
            pixels = pixels[:, :, ::-1]
        return np.ascontiguousarray(pixels)

    def convert_img(self, img: np.ndarray, origin: type) -> np.ndarray:
        """Return packed RGB. The native engine swaps that to planar BGR.

        3.x returns packed BGR. A path/bytes/PIL image is RGB from Pillow, and
        3.x swaps it to BGR. An ndarray is left as BGR (OpenCV). Doing the
        opposite swap here feeds the native engine the same colors.
        """

        if img.dtype == np.bool_ or str(img.dtype) == "bool":
            img = img.astype(np.uint8) * 255
        if img.ndim == 2:
            plane = img
            rgb = np.stack([plane, plane, plane], axis=-1)
            return self._as_uint8(rgb)
        if img.ndim != 3:
            raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")
        channels = img.shape[2]
        from_array = issubclass(origin, np.ndarray)
        if channels == 1:
            plane = img[:, :, 0]
            return self._as_uint8(np.stack([plane, plane, plane], axis=-1))
        if channels == 2:
            return self._as_uint8(_two_channel_to_rgb(img))
        if channels == 4:
            return self._as_uint8(_rgba_to_rgb(img))
        if channels != 3:
            raise LoadImageError(f"The channel({channels}) of the img is not in [1, 2, 3, 4]")
        rgb = img[:, :, ::-1] if from_array else img
        return np.ascontiguousarray(self._as_uint8(rgb))

    @staticmethod
    def _as_uint8(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return np.ascontiguousarray(img)
        return np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))

    def _decode_bytes(self, data: bytes) -> np.ndarray:
        try:
            image = Image.open(BytesIO(data))
            if image.mode == "1":
                image = image.convert("L")
            image = ImageOps.exif_transpose(image) or image
            return np.asarray(image)
        except UnidentifiedImageError as exc:
            raise LoadImageError("cannot identify image file") from exc

    def _load_memory(self, img: MemoryImage) -> np.ndarray:
        if img.address <= 0 or img.length <= 0:
            raise LoadImageError("MemoryImage loading failed.")
        encoded = ctypes.string_at(img.address, img.length)
        try:
            return self._decode_bytes(encoded)
        except LoadImageError as exc:
            raise LoadImageError("MemoryImage loading failed.") from exc


def _two_channel_to_rgb(img: np.ndarray) -> np.ndarray:
    """Gray+alpha composite used by 3.x ``cvt_two_to_three`` (channels are equal)."""

    gray = img[..., 0]
    alpha = img[..., 1]
    kept = np.where(alpha[..., None] > 0, np.stack([gray, gray, gray], axis=-1), 0)
    not_a = np.bitwise_not(alpha.astype(np.uint8))
    background = np.stack([not_a, not_a, not_a], axis=-1)
    return np.clip(kept.astype(np.int16) + background.astype(np.int16), 0, 255).astype(np.uint8)


def _rgba_to_rgb(img: np.ndarray) -> np.ndarray:
    """Contrast background from 3.x, returned as RGB for the native engine."""

    rgb = img[:, :, :3].astype(np.uint8)
    alpha = img[:, :, 3]
    mask = alpha > 0
    opaque = rgb[mask]
    if opaque.size == 0:
        bg = np.array([255, 255, 255], dtype=np.float32)
    else:
        luminance = 0.299 * opaque[:, 0] + 0.587 * opaque[:, 1] + 0.114 * opaque[:, 2]
        bg = np.array([255, 255, 255] if float(np.mean(luminance)) < 128 else [0, 0, 0], dtype=np.float32)
    alpha_norm = alpha.astype(np.float32) / 255.0
    blended = rgb.astype(np.float32) * alpha_norm[..., None] + bg * (1.0 - alpha_norm[..., None])
    return np.clip(blended, 0, 255).astype(np.uint8)
