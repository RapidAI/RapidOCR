# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import Tuple

import cv2
import numpy as np


def reduce_max_side(
    img: np.ndarray, max_side_len: int = 2000
) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]

    ratio = 1.0
    if max(h, w) > max_side_len:
        if h > w:
            ratio = float(max_side_len) / h
        else:
            ratio = float(max_side_len) / w

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = int(round(resize_h / 32) * 32)
    resize_w = int(round(resize_w / 32) * 32)

    try:
        if int(resize_w) <= 0 or int(resize_h) <= 0:
            raise ResizeImgError("resize_w or resize_h is less than or equal to 0")
        img = cv2.resize(img, (resize_w, resize_h))
    except Exception as exc:
        raise ResizeImgError() from exc

    ratio_h = h / resize_h
    ratio_w = w / resize_w
    return img, ratio_h, ratio_w


def increase_min_side(
    img: np.ndarray, min_side_len: int = 30
) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]

    ratio = 1.0
    if min(h, w) < min_side_len:
        if h < w:
            ratio = float(min_side_len) / h
        else:
            ratio = float(min_side_len) / w

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = int(round(resize_h / 32) * 32)
    resize_w = int(round(resize_w / 32) * 32)

    try:
        if int(resize_w) <= 0 or int(resize_h) <= 0:
            raise ResizeImgError("resize_w or resize_h is less than or equal to 0")
        img = cv2.resize(img, (resize_w, resize_h))
    except Exception as exc:
        raise ResizeImgError() from exc

    ratio_h = h / resize_h
    ratio_w = w / resize_w
    return img, ratio_h, ratio_w


def add_round_letterbox(
    img: np.ndarray,
    padding_tuple: Tuple[int, int, int, int],
) -> np.ndarray:
    padded_img = cv2.copyMakeBorder(
        img,
        padding_tuple[0],
        padding_tuple[1],
        padding_tuple[2],
        padding_tuple[3],
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded_img


class ResizeImgError(Exception):
    pass
