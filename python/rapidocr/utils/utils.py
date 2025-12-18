# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import hashlib
import importlib
from pathlib import Path
from sys import platform
from typing import Any, List, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np


def filter_by_indices(
    data: Union[np.ndarray, List[Any], Tuple[Any]],
    indices: Union[np.ndarray, List[int], Tuple[int, ...]],
) -> Union[np.ndarray, List[Any], Tuple[Any]]:
    if isinstance(data, np.ndarray):
        return data[indices]

    if isinstance(data, (list, tuple)):
        return [data[i] for i in indices]

    raise TypeError(f"Unsupported data type: {type(data)}")


def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def quads_to_rect_bbox(bbox: np.ndarray) -> Tuple[float, float, float, float]:
    if bbox.ndim != 3:
        raise ValueError("bbox shape must be 3")

    if bbox.shape[1] != 4 and bbox.shape[2] != 2:
        raise ValueError("bbox shape must be (N, 4, 2)")

    all_x, all_y = (bbox[:, :, 0].flatten(), bbox[:, :, 1].flatten())
    x_min, y_min = np.min(all_x), np.min(all_y)
    x_max, y_max = np.max(all_x), np.max(all_y)
    return float(x_min), float(y_min), float(x_max), float(y_max)


def is_chinese_char(ch: str) -> bool:
    return (
        "\u4e00" <= ch <= "\u9fff"  # 汉字
        or "\u3000" <= ch <= "\u303f"  # CJK 标点（如 。 、 “” 《》 ……）
        or "\uff00" <= ch <= "\uffef"  # 全角符号（如 ，．！？【】）
    )


def has_chinese_char(text: str) -> bool:
    return any(is_chinese_char(ch) for ch in text)


def get_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    with open(file_path, "rb") as file:
        sha_signature = hashlib.sha256()
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            sha_signature.update(chunk)

    return sha_signature.hexdigest()


def save_img(save_path: Union[str, Path], img: np.ndarray):
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if platform.startswith("win32"):
        cv2.imencode(Path(save_path).suffix, img)[1].tofile(str(save_path))
        return

    cv2.imwrite(str(save_path), img)


def is_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception as e:
        return False


def import_package(name, package=None):
    try:
        module = importlib.import_module(name, package=package)
        return module
    except ModuleNotFoundError:
        return None
