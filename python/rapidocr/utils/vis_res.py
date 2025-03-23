# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from .load_image import LoadImage
from .logger import Logger
from .utils import download_file

root_dir = Path(__file__).resolve().parent.parent
InputType = Union[str, np.ndarray, bytes, Path, Image.Image]

DEFAULT_FONT_DIR = root_dir / "models"
DEFAULT_FONT_PATH = DEFAULT_FONT_DIR / "FZYTK.TTF"
FONT_YAML_PATH = root_dir / "default_models.yaml"


class VisRes:
    def __init__(self, text_score: float = 0.5):
        self.logger = Logger(logger_name=__name__).get_log()

        self.text_score = text_score
        self.load_img = LoadImage()

        self.font_cfg = OmegaConf.load(FONT_YAML_PATH)["fonts"]

    def __call__(
        self,
        img_content: InputType,
        dt_boxes: np.ndarray,
        txts: Optional[Union[List[str], Tuple[str]]] = None,
        scores: Optional[Union[Tuple[float], List[float]]] = None,
        font_path: Optional[str] = None,
        lang_rec: Optional[str] = None,
    ) -> np.ndarray:
        if txts is None:
            return self.draw_dt_boxes(img_content, dt_boxes, scores)

        font_path = self.get_font_path(font_path, lang_rec)
        return self.draw_ocr_box_txt(img_content, dt_boxes, txts, scores, font_path)

    def draw_dt_boxes(
        self, img_content: InputType, dt_boxes: np.ndarray, scores: List[float]
    ) -> np.ndarray:
        img = self.load_img(img_content)

        for idx, (box, score) in enumerate(zip(dt_boxes, scores)):
            color = self.get_random_color()

            points = np.array(box)
            cv2.polylines(img, np.int32([points]), 1, color=color, thickness=1)

            start_point = round(points[0][0]), round(points[0][1])
            cv2.putText(
                img,
                f"{idx}:{score:.3f}",
                start_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3,
            )
        return img

    def get_font_path(
        self,
        font_path: Optional[Union[str, Path]] = None,
        lang_rec: Optional[str] = None,
    ) -> str:
        if lang_rec is None:
            # 没有指定语种，用默认字体文件
            download_file(self.font_cfg["ch"], DEFAULT_FONT_PATH, logger=self.logger)
            return str(DEFAULT_FONT_PATH)

        if font_path is None:
            # 指定了语种，但是没有指定字体文件，根据语种选择字体文件
            lang_rec = lang_rec.rsplit("_", 1)[0]
            font_url = self.font_cfg.get(lang_rec, None)
            if font_url is None:
                self.logger.warning(
                    "Font file for %s is not found in the supported font list. Default font file will be used.",
                    lang_rec,
                )
                download_file(
                    self.font_cfg["ch"], DEFAULT_FONT_PATH, logger=self.logger
                )
                return str(DEFAULT_FONT_PATH)

            save_font_path = DEFAULT_FONT_DIR / f"{Path(font_url).name}"
            download_file(font_url, save_font_path, logger=self.logger)
            return str(save_font_path)

        return str(font_path)

    def draw_rec_res(
        self,
        imgs: List[InputType],
        txts: Union[List[str], Tuple[str]],
        scores: Optional[Tuple[float]] = None,
    ) -> np.ndarray:
        print("ok")

    def draw_ocr_box_txt(
        self,
        img_content: InputType,
        dt_boxes: np.ndarray,
        txts: Union[List[str], Tuple[str]],
        scores: Optional[Tuple[float]] = None,
        font_path: Optional[str] = None,
    ) -> np.ndarray:
        image = Image.fromarray(self.load_img(img_content))
        h, w = image.height, image.width
        if image.mode == "L":
            image = image.convert("RGB")

        img_left = image.copy()
        img_right = Image.new("RGB", (w, h), (255, 255, 255))

        random.seed(0)
        draw_left = ImageDraw.Draw(img_left)
        draw_right = ImageDraw.Draw(img_right)
        for idx, (box, txt) in enumerate(zip(dt_boxes, txts)):
            if scores is not None and float(scores[idx]) < self.text_score:
                continue

            color = self.get_random_color()

            box_list = np.array(box).reshape(8).tolist()
            draw_left.polygon(box_list, fill=color)
            draw_right.polygon(box_list, outline=color)

            box_height = self.get_box_height(box)
            box_width = self.get_box_width(box)
            if box_height > 2 * box_width:
                font_size = max(int(box_width * 0.9), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                cur_y = box[0][1]

                for c in txt:
                    draw_right.text(
                        (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font
                    )
                    cur_y += self.get_char_size(font, c)
            else:
                font_size = max(int(box_height * 0.8), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                draw_right.text([box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)

        img_left = Image.blend(image, img_left, 0.5)
        img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(img_right, (w, 0, w * 2, h))
        return np.array(img_show)

    @staticmethod
    def get_random_color() -> Tuple[int, int, int]:
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

    @staticmethod
    def get_box_height(box: List[List[float]]) -> float:
        return math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)

    @staticmethod
    def get_box_width(box: List[List[float]]) -> float:
        return math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)

    @staticmethod
    def get_char_size(font, char_str: str) -> float:
        # compatible with Pillow v9 and v10.
        if hasattr(font, "getsize"):
            get_size_func = getattr(font, "getsize")
            return get_size_func(char_str)[1]

        if hasattr(font, "getlength"):
            get_size_func = getattr(font, "getlength")
            return get_size_func(char_str)

        raise ValueError(
            "The Pillow ImageFont instance has not getsize or getlength func."
        )
