# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def map_img_to_original(
    imgs: List[np.ndarray], ratio_h: float, ratio_w: float
) -> List[np.ndarray]:
    results = []
    for img in imgs:
        img_h, img_w = img.shape[:2]
        ori_img_h, ori_img_w = round(img_h * ratio_h), round(img_w * ratio_w)
        resize_img = cv2.resize(img, (ori_img_w, ori_img_h))
        results.append(resize_img)
    return results


def map_boxes_to_original(
    dt_boxes: np.ndarray, op_record: Dict[str, Any], ori_h: int, ori_w: int
) -> np.ndarray:
    for op in reversed(list(op_record.keys())):
        v = op_record[op]
        if "padding" in op:
            top, left = v.get("top"), v.get("left")
            dt_boxes[:, :, 0] -= left
            dt_boxes[:, :, 1] -= top
        elif "preprocess" in op:
            ratio_h = v.get("ratio_h")
            ratio_w = v.get("ratio_w")
            dt_boxes[:, :, 0] *= ratio_w
            dt_boxes[:, :, 1] *= ratio_h

    dt_boxes = np.where(dt_boxes < 0, 0, dt_boxes)
    dt_boxes[..., 0] = np.where(dt_boxes[..., 0] > ori_w, ori_w, dt_boxes[..., 0])
    dt_boxes[..., 1] = np.where(dt_boxes[..., 1] > ori_h, ori_h, dt_boxes[..., 1])
    return dt_boxes


def apply_vertical_padding(
    img: np.ndarray,
    op_record: Dict[str, Any],
    width_height_ratio: float,
    min_height: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = img.shape[:2]

    if width_height_ratio == -1:
        use_limit_ratio = False
    else:
        use_limit_ratio = w / h > width_height_ratio

    if h <= min_height or use_limit_ratio:
        padding_h = get_padding_h(h, w, width_height_ratio, min_height)
        block_img = add_round_letterbox(img, (padding_h, padding_h, 0, 0))
        op_record["padding_1"] = {"top": padding_h, "left": 0}
        return block_img, op_record

    op_record["padding_1"] = {"top": 0, "left": 0}
    return img, op_record


def get_padding_h(h: int, w: int, width_height_ratio: float, min_height: float) -> int:
    new_h = max(int(w / width_height_ratio), min_height) * 2
    padding_h = int(abs(new_h - h) / 2)
    return padding_h


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    pts_std = np.array(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    ).astype(np.float32)
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def resize_image_within_bounds(
    img: np.ndarray, min_side_len: float, max_side_len: float
) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    max_value = max(h, w)
    ratio_h = ratio_w = 1.0
    if max_value > max_side_len:
        img, ratio_h, ratio_w = reduce_max_side(img, max_side_len)

    h, w = img.shape[:2]
    min_value = min(h, w)
    if min_value < min_side_len:
        img, ratio_h, ratio_w = increase_min_side(img, min_side_len)
    return img, ratio_h, ratio_w


def reduce_max_side(
    img: np.ndarray, max_side_len: float = 2000
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
    img: np.ndarray, min_side_len: float = 30
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
    img: np.ndarray, padding_tuple: Tuple[int, int, int, int]
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
