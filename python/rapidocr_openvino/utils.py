# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
from io import BytesIO
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import yaml
from openvino.runtime import Core
from PIL import Image, UnidentifiedImageError

root_dir = Path(__file__).resolve().parent
InputType = Union[str, np.ndarray, bytes, Path]


class OpenVINOInferSession:
    def __init__(self, config):
        ie = Core()

        config["model_path"] = str(root_dir / config["model_path"])
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

        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3 and img.shape[2] == 4:
            return self.cvt_four_to_three(img)

        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = np.array(Image.open(img))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = np.array(Image.open(BytesIO(img)))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        if isinstance(img, np.ndarray):
            return img

        raise LoadImageError(f"{type(img)} is not supported!")

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → RGB"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)
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
    global_group.add_argument("--use_angle_cls", type=bool, default=True)
    global_group.add_argument("--use_text_det", type=bool, default=True)
    global_group.add_argument("--print_verbose", type=bool, default=False)
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
    det_group.add_argument("--det_use_dilation", type=bool, default=True)
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
    rec_group.add_argument("--rec_image_shape", type=list, default=[3, 48, 320])
    rec_group.add_argument("--rec_batch_num", type=int, default=6)

    args = parser.parse_args()
    return args


class UpdateParameters:
    def __init__(self) -> None:
        pass

    def parse_kwargs(self, **kwargs):
        global_dict, det_dict, cls_dict, rec_dict = {}, {}, {}, {}
        for k, v in kwargs.items():
            if k.startswith("det"):
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
            "Det": self.update_det_params(config["Det"], det_dict),
            "Cls": self.update_cls_params(config["Cls"], cls_dict),
            "Rec": self.update_rec_params(config["Rec"], rec_dict),
        }
        return new_config

    def update_global_params(self, config, global_dict):
        if global_dict:
            config.update(global_dict)
        return config

    def update_det_params(self, config, det_dict):
        if det_dict:
            det_dict = {k.split("det_")[1]: v for k, v in det_dict.items()}
            if not det_dict["model_path"]:
                det_dict["model_path"] = str(root_dir / config["model_path"])
            config.update(det_dict)
        return config

    def update_cls_params(self, config, cls_dict):
        if cls_dict:
            need_remove_prefix = ["cls_label_list", "cls_model_path"]
            new_cls_dict = {}
            for k, v in cls_dict.items():
                if k in need_remove_prefix:
                    k = k.split("cls_")[1]
                new_cls_dict[k] = v

            if not new_cls_dict["model_path"]:
                new_cls_dict["model_path"] = str(root_dir / config["model_path"])
            config.update(new_cls_dict)
        return config

    def update_rec_params(self, config, rec_dict):
        if rec_dict:
            need_remove_prefix = ["rec_model_path"]
            new_rec_dict = {}
            for k, v in rec_dict.items():
                if k in need_remove_prefix:
                    k = k.split("rec_")[1]
                new_rec_dict[k] = v

            if not new_rec_dict["model_path"]:
                new_rec_dict["model_path"] = str(root_dir / config["model_path"])
            config.update(new_rec_dict)
        return config
