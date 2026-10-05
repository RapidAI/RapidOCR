"""Side-length resize and vertical padding from rapidocr 3.x, without OpenCV."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


class ResizeImgError(Exception):
    pass


def resize_image_within_bounds(
    img: np.ndarray, min_side_len: float, max_side_len: float
) -> Tuple[np.ndarray, float, float]:
    height, width = img.shape[:2]
    ratio_h = ratio_w = 1.0
    if max(height, width) > max_side_len:
        img, ratio_h, ratio_w = reduce_max_side(img, max_side_len)
    height, width = img.shape[:2]
    if min(height, width) < min_side_len:
        img, ratio_h, ratio_w = increase_min_side(img, min_side_len)
    return img, ratio_h, ratio_w


def reduce_max_side(img: np.ndarray, max_side_len: float = 2000) -> Tuple[np.ndarray, float, float]:
    return _resize_to_ratio(img, max_side_len, grow=False)


def increase_min_side(img: np.ndarray, min_side_len: float = 30) -> Tuple[np.ndarray, float, float]:
    return _resize_to_ratio(img, min_side_len, grow=True)


def _resize_to_ratio(img: np.ndarray, side: float, grow: bool) -> Tuple[np.ndarray, float, float]:
    height, width = img.shape[:2]
    ratio = 1.0
    if grow:
        if min(height, width) < side:
            ratio = float(side) / (height if height < width else width)
    elif max(height, width) > side:
        ratio = float(side) / (height if height > width else width)
    resize_h = int(round(int(height * ratio) / 32) * 32)
    resize_w = int(round(int(width * ratio) / 32) * 32)
    if resize_w <= 0 or resize_h <= 0:
        raise ResizeImgError("resize_w or resize_h is less than or equal to 0")
    if resize_h == height and resize_w == width:
        return img, 1.0, 1.0
    resized = np.asarray(Image.fromarray(img).resize((resize_w, resize_h), Image.BILINEAR))
    return resized, height / resize_h, width / resize_w


def apply_vertical_padding(
    img: np.ndarray, width_height_ratio: float, min_height: float
) -> Tuple[np.ndarray, int]:
    """Return the padded image and the top padding, matching 3.x."""

    height, width = img.shape[:2]
    use_limit_ratio = False if width_height_ratio == -1 else width / max(height, 1) > width_height_ratio
    if height <= min_height or use_limit_ratio:
        new_h = max(int(width / width_height_ratio), int(min_height)) * 2
        padding_h = int(abs(new_h - height) / 2)
        if padding_h <= 0:
            return img, 0
        padded = np.pad(img, ((padding_h, padding_h), (0, 0), (0, 0)), mode="constant")
        return padded, padding_h
    return img, 0


def map_boxes_to_original(
    boxes: np.ndarray,
    ratio_w: float,
    ratio_h: float,
    pad_top: int,
    ori_w: int,
    ori_h: int,
) -> np.ndarray:
    mapped = boxes.astype(np.float32, copy=True)
    mapped[:, :, 1] -= pad_top
    mapped[:, :, 0] *= ratio_w
    mapped[:, :, 1] *= ratio_h
    mapped[:, :, 0] = np.clip(mapped[:, :, 0], 0, ori_w)
    mapped[:, :, 1] = np.clip(mapped[:, :, 1], 0, ori_h)
    return mapped
