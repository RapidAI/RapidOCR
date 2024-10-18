# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
InputType = Union[str, np.ndarray, bytes, Path, Image.Image]


def update_model_path(config: Dict[str, Any]) -> Dict[str, Any]:
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
    global_group.add_argument("--max_side_len", type=int, default=2000)
    global_group.add_argument("--min_side_len", type=int, default=30)

    global_group.add_argument("--inference_num_threads", type=int, default=-1)

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
    rec_group.add_argument("--rec_keys_path", type=str, default=None)
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
            "Det": self.update_params(
                config["Det"],
                det_dict,
                "det_",
                ["det_model_path"],
            ),
            "Cls": self.update_params(
                config["Cls"],
                cls_dict,
                "cls_",
                ["cls_label_list", "cls_model_path"],
            ),
            "Rec": self.update_params(
                config["Rec"],
                rec_dict,
                "rec_",
                ["rec_model_path"],
            ),
        }

        update_params = ["inference_num_threads"]
        new_config = self.update_global_to_module(
            config, update_params, src="Global", dsts=["Det", "Cls", "Rec"]
        )
        return new_config

    def update_global_to_module(
        self, config, params: List[str], src: str, dsts: List[str]
    ):
        for dst in dsts:
            for param in params:
                config[dst].update({param: config[src][param]})
        return config

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
