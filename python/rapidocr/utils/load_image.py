# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from io import BytesIO
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps, UnidentifiedImageError

from .utils import is_url

root_dir = Path(__file__).resolve().parent
InputType = Union[str, np.ndarray, bytes, Path, Image.Image]


class LoadImage:
    def __init__(self):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        origin_img_type = type(img)
        img = self.load_img(img)
        img = self.convert_img(img, origin_img_type)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            if is_url(str(img)):
                img = Image.open(requests.get(img, stream=True, timeout=60).raw)
            else:
                self.verify_exist(img)
                img = Image.open(img)

            img = self.exif_transpose(img)

            try:
                img = self.img_to_ndarray(img)
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = self.img_to_ndarray(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        if isinstance(img, Image.Image):
            return self.img_to_ndarray(img)

        raise LoadImageError(f"{type(img)} is not supported!")

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")

    @staticmethod
    def exif_transpose(img: Image.Image) -> Image.Image:
        try:
            img_corrected = ImageOps.exif_transpose(img)
            if img_corrected is None:
                return img
            return img_corrected
        except Exception as e:
            return img

    def img_to_ndarray(self, img: Image.Image) -> np.ndarray:
        if img.mode == "1":
            img = img.convert("L")
            return np.array(img)
        return np.array(img)

    def convert_img(self, img: np.ndarray, origin_img_type: Any) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 3:
                if issubclass(origin_img_type, (str, Path, bytes, Image.Image)):
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img

            if channel == 4:
                return self.cvt_four_to_three(img)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR"""
        img_gray = img[..., 0]
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_alpha = img[..., 1]
        not_a = cv2.bitwise_not(img_alpha)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """自动调整背景颜色，以增强文字对比度"""

        rgb = img[:, :, :3]  # shape (H, W, 3)
        alpha = img[:, :, 3]  # shape (H, W)

        # 获取非透明区域的 RGB 像素
        mask = alpha > 0
        non_transparent_rgb = rgb[mask]  # shape (N, 3)
        if non_transparent_rgb.size == 0:
            # 全透明图像：默认用白色背景
            bg_color = (255, 255, 255)
        else:
            # 使用加权灰度公式计算亮度均值
            # luminance = 0.299*R + 0.587*G + 0.114*B
            r, g, b = (
                non_transparent_rgb[:, 0],
                non_transparent_rgb[:, 1],
                non_transparent_rgb[:, 2],
            )
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            avg_luminance = np.mean(luminance)

            # 根据平均亮度选择高对比度背景
            bg_color = (255, 255, 255) if avg_luminance < 128 else (0, 0, 0)

        # 构建背景图像
        background = np.full_like(rgb, bg_color, dtype=np.uint8)

        # 合成：前景 = rgb * (alpha/255), 背景 = bg * (1 - alpha/255)
        alpha_norm = alpha.astype(np.float32) / 255.0
        foreground_blend = rgb.astype(np.float32) * alpha_norm[..., None]
        background_blend = background.astype(np.float32) * (1.0 - alpha_norm)[..., None]

        blended = (foreground_blend + background_blend).astype(np.uint8)

        return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)


class LoadImageError(Exception):
    pass
