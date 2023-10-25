# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import math
import random
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from openvino.runtime import Core
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

root_dir = Path(__file__).resolve().parent
InputType = Union[str, np.ndarray, bytes, Path]


class OpenVINOInferSession:
    def __init__(self, config):
        ie = Core()

        self._verify_model(config["model_path"])
        model_onnx = ie.read_model(config["model_path"])
        compile_model = ie.compile_model(model=model_onnx, device_name="CPU")
        self.session = compile_model.create_infer_request()

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        self.session.infer(inputs=[input_content])
        return self.session.get_output_tensor().data

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


class LoadImage:
    def __init__(
        self,
    ):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        img = self.load_img(img)
        img = self.convert_img(img)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = np.array(Image.open(img))
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = np.array(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        raise LoadImageError(f"{type(img)} is not supported!")

    def convert_img(self, img: np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 4:
                return self.cvt_four_to_three(img)

            if channel == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → BGR"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)
        new_img = cv2.add(new_img, not_a)
        return new_img

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
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class LoadImageError(Exception):
    pass


def read_yaml(yaml_path):
    with open(yaml_path, "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


def concat_model_path(config):
    key = "model_path"
    config["Det"][key] = str(root_dir / config["Det"][key])
    config["Rec"][key] = str(root_dir / config["Rec"][key])
    config["Cls"][key] = str(root_dir / config["Cls"][key])
    return config


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", "--img_path", type=str, default=None, required=True)
    parser.add_argument("-p", "--print_cost", action="store_true", default=False)

    global_group = parser.add_argument_group(title="Global")
    global_group.add_argument("--text_score", type=float, default=0.5)

    global_group.add_argument("--no_det", action="store_true", default=False)
    global_group.add_argument("--no_cls", action="store_true", default=False)
    global_group.add_argument("--no_rec", action="store_true", default=False)

    global_group.add_argument("--print_verbose", action="store_true", default=False)
    global_group.add_argument("--min_height", type=int, default=30)
    global_group.add_argument("--width_height_ratio", type=int, default=8)

    det_group = parser.add_argument_group(title="Det")
    det_group.add_argument("--det_model_path", type=str, default=None)
    det_group.add_argument("--det_limit_side_len", type=float, default=736)
    det_group.add_argument(
        "--det_limit_type", type=str, default="min", choices=["max", "min"]
    )
    det_group.add_argument("--det_thresh", type=float, default=0.3)
    det_group.add_argument("--det_box_thresh", type=float, default=0.5)
    det_group.add_argument("--det_unclip_ratio", type=float, default=1.6)
    det_group.add_argument(
        "--det_donot_use_dilation", action="store_true", default=False
    )
    det_group.add_argument(
        "--det_score_mode", type=str, default="fast", choices=["slow", "fast"]
    )

    cls_group = parser.add_argument_group(title="Cls")
    cls_group.add_argument("--cls_model_path", type=str, default=None)
    cls_group.add_argument("--cls_image_shape", type=list, default=[3, 48, 192])
    cls_group.add_argument("--cls_label_list", type=list, default=["0", "180"])
    cls_group.add_argument("--cls_batch_num", type=int, default=6)
    cls_group.add_argument("--cls_thresh", type=float, default=0.9)

    rec_group = parser.add_argument_group(title="Rec")
    rec_group.add_argument("--rec_model_path", type=str, default=None)
    rec_group.add_argument("--rec_img_shape", type=list, default=[3, 48, 320])
    rec_group.add_argument("--rec_batch_num", type=int, default=6)

    vis_group = parser.add_argument_group(title="Visual Result")
    vis_group.add_argument("-vis", "--vis_res", action="store_true", default=False)
    vis_group.add_argument(
        "--vis_font_path",
        type=str,
        default=None,
        help="When -vis is True, the font_path must have value.",
    )
    vis_group.add_argument(
        "--vis_save_path",
        type=str,
        default=".",
        help="The directory of saving the vis image.",
    )

    args = parser.parse_args()
    return args


class UpdateParameters:
    def __init__(self) -> None:
        pass

    def parse_kwargs(self, **kwargs):
        global_dict, det_dict, cls_dict, rec_dict = {}, {}, {}, {}
        for k, v in kwargs.items():
            if k.startswith("det"):
                k = k.split("det_")[1]
                if k == "donot_use_dilation":
                    k = "use_dilation"
                    v = not v

                det_dict[k] = v
            elif k.startswith("cls"):
                cls_dict[k] = v
            elif k.startswith("rec"):
                rec_dict[k] = v
            else:
                global_dict[k] = v
        return global_dict, det_dict, cls_dict, rec_dict

    def __call__(self, config, **kwargs):
        global_dict, det_dict, cls_dict, rec_dict = self.parse_kwargs(**kwargs)
        new_config = {
            "Global": self.update_global_params(config["Global"], global_dict),
            "Det": self.update_params(config["Det"], det_dict, "det_", None),
            "Cls": self.update_params(
                config["Cls"],
                cls_dict,
                "cls_",
                ["cls_label_list", "cls_model_path", "cls_use_cuda"],
            ),
            "Rec": self.update_params(
                config["Rec"], rec_dict, "rec_", ["rec_model_path", "rec_use_cuda"]
            ),
        }
        return new_config

    def update_global_params(self, config, global_dict):
        if global_dict:
            config.update(global_dict)
        return config

    def update_params(
        self,
        config,
        param_dict: Dict[str, str],
        prefix: str,
        need_remove_prefix: Optional[List[str]] = None,
    ):
        if not param_dict:
            return config

        filter_dict = self.remove_prefix(param_dict, prefix, need_remove_prefix)
        model_path = filter_dict.get("model_path", None)
        if not model_path:
            filter_dict["model_path"] = str(root_dir / config["model_path"])

        config.update(filter_dict)
        return config

    @staticmethod
    def remove_prefix(
        config: Dict[str, str],
        prefix: str,
        need_remove_prefix: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        if not need_remove_prefix:
            return config

        new_rec_dict = {}
        for k, v in config.items():
            if k in need_remove_prefix:
                k = k.split(prefix)[1]
            new_rec_dict[k] = v
        return new_rec_dict


class VisRes:
    def __init__(self, text_score: float = 0.5):
        self.text_score = text_score
        self.load_img = LoadImage()

    def __call__(
        self,
        img_content: InputType,
        dt_boxes: np.ndarray,
        txts: Optional[Union[List[str], Tuple[str]]] = None,
        scores: Optional[Tuple[float]] = None,
        font_path: Optional[str] = None,
    ) -> np.ndarray:
        if txts is None and scores is None:
            return self.draw_dt_boxes(img_content, dt_boxes)
        return self.draw_ocr_box_txt(img_content, dt_boxes, txts, scores, font_path)

    def draw_dt_boxes(self, img_content: InputType, dt_boxes: np.ndarray) -> np.ndarray:
        img = self.load_img(img_content)

        for idx, box in enumerate(dt_boxes):
            color = self.get_random_color()

            points = np.array(box)
            cv2.polylines(img, np.int32([points]), 1, color=color, thickness=1)

            start_point = round(points[0][0]), round(points[0][1])
            cv2.putText(
                img, f"{idx}", start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3
            )
        return img

    def draw_ocr_box_txt(
        self,
        img_content: InputType,
        dt_boxes: np.ndarray,
        txts: Optional[Union[List[str], Tuple[str]]] = None,
        scores: Optional[Tuple[float]] = None,
        font_path: Optional[str] = None,
    ) -> np.ndarray:
        font_path = self.get_font_path(font_path)

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
                    char_size = font.getsize(c)
                    draw_right.text(
                        (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font
                    )
                    cur_y += char_size[1]
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
    def get_font_path(font_path: Optional[Union[str, Path]] = None) -> str:
        if font_path is None or not Path(font_path).exists():
            raise FileNotFoundError(
                f"The {font_path} does not exists! \n"
                f"You could download the file in the https://drive.google.com/file/d/1evWVX38EFNwTq_n5gTFgnlv8tdaNcyIA/view?usp=sharing"
            )
        return str(font_path)

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
