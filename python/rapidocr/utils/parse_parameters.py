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
    elif lang in ["ch", "en"]:
        pass
    else:
        raise ValueError(f"lang: {lang} is not in the supported list.")

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
    def update_model_path(cls, cfg: DictConfig) -> DictConfig:
        keys = ["model_path", "model_dir"]
        for key in keys:
            cls.update(cfg, f"Det.{key}", str(root_dir / cfg.Det[key]))
            cls.update(cfg, f"Cls.{key}", str(root_dir / cfg.Cls[key]))
            cls.update(cfg, f"Rec.{key}", str(root_dir / cfg.Rec[key]))
        return cfg

    @classmethod
    def update_dict_path(
        cls, cfg: DictConfig, key: str = "rec_keys_path"
    ) -> DictConfig:
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
    parser.add_argument("--text_score", type=float, default=0.5)
    parser.add_argument("-vis", "--vis_res", action="store_true", default=False)
    parser.add_argument(
        "--vis_font_path",
        type=str,
        default=None,
        help="When -vis is True, the font_path must have value.",
    )
    parser.add_argument(
        "--vis_save_path",
        type=str,
        default=".",
        help="The directory of saving the vis image.",
    )

    args = parser.parse_args()
    return args
