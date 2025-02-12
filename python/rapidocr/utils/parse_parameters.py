# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
from pathlib import Path
from typing import Dict, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
InputType = Union[str, np.ndarray, bytes, Path, Image.Image]


def parse_lang(lang):
    latin_lang = [
        "af",
        "az",
        "bs",
        "cs",
        "cy",
        "da",
        "de",
        "es",
        "et",
        "fr",
        "ga",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
        "ro",
        "rs_latin",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
        "french",
        "german",
    ]
    arabic_lang = ["ar", "fa", "ug", "ur"]
    cyrillic_lang = [
        "ru",
        "rs_cyrillic",
        "be",
        "bg",
        "uk",
        "mn",
        "abq",
        "ady",
        "kbd",
        "ava",
        "dar",
        "inh",
        "che",
        "lbe",
        "lez",
        "tab",
    ]
    devanagari_lang = [
        "hi",
        "mr",
        "ne",
        "bh",
        "mai",
        "ang",
        "bho",
        "mah",
        "sck",
        "new",
        "gom",
        "sa",
        "bgc",
    ]
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"

    if lang == "ch":
        det_lang = "ch"
    elif lang in ["en", "latin"]:
        det_lang = "en"
    else:
        det_lang = "Multilingual"

    return det_lang, lang


class ParseParams(OmegaConf):
    def __init__(self):
        pass

    @classmethod
    def update_model_path(cls, cfg: DictConfig, key: str = "model_path") -> DictConfig:
        cls.update(cfg, f"Det.{key}", str(root_dir / cfg.Det[key]))
        cls.update(cfg, f"Cls.{key}", str(root_dir / cfg.Cls[key]))
        cls.update(cfg, f"Rec.{key}", str(root_dir / cfg.Rec[key]))
        return cfg

    @classmethod
    def update_batch(cls, cfg: DictConfig, params: Dict[str, str]) -> DictConfig:
        for k, v in params.items():
            cls.update(cfg, k, v)
        return cfg


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
    global_group.add_argument("--return_word_box", action="store_true", default=False)

    global_group.add_argument("--intra_op_num_threads", type=int, default=-1)
    global_group.add_argument("--inter_op_num_threads", type=int, default=-1)

    det_group = parser.add_argument_group(title="Det")
    det_group.add_argument("--det_use_cuda", action="store_true", default=False)
    det_group.add_argument("--det_use_dml", action="store_true", default=False)
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
    cls_group.add_argument("--cls_use_cuda", action="store_true", default=False)
    cls_group.add_argument("--cls_use_dml", action="store_true", default=False)
    cls_group.add_argument("--cls_model_path", type=str, default=None)
    cls_group.add_argument("--cls_image_shape", type=list, default=[3, 48, 192])
    cls_group.add_argument("--cls_label_list", type=list, default=["0", "180"])
    cls_group.add_argument("--cls_batch_num", type=int, default=6)
    cls_group.add_argument("--cls_thresh", type=float, default=0.9)

    rec_group = parser.add_argument_group(title="Rec")
    rec_group.add_argument("--rec_use_cuda", action="store_true", default=False)
    rec_group.add_argument("--rec_use_dml", action="store_true", default=False)
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
