# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
InputType = Union[str, np.ndarray, bytes, Path, Image.Image]


class ParseLang:
    def __init__(self):
        self.latin_lang = [
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

        self.arabic_lang = ["ar", "fa", "ug", "ur"]
        self.cyrillic_lang = [
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
        self.devanagari_lang = [
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

    def __call__(self, lang_det: str, lang_rec: str) -> Tuple[str, str]:
        lang_det, det_model_type = lang_det.split("_")
        lang_rec, rec_model_type = lang_rec.split("_")
        return f"{lang_det}_{det_model_type}", f"{lang_rec}_{rec_model_type}"

    def parse_det_lang(self, lang: str) -> str:
        if lang == "ch":
            return "ch"

        if lang in ["en", "latin"]:
            return "en"

        return "Multilingual"

    def parse_rec_lang(self, lang: str):
        if lang in self.latin_lang:
            return "latin"

        if lang in self.arabic_lang:
            return "arabic"

        if lang in self.cyrillic_lang:
            return "cyrillic"

        if lang in self.devanagari_lang:
            return "devanagari"

        if lang in ["ch", "en"]:
            return

        raise ValueError(f"lang: {lang} is not in the supported list.")


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
    def update_batch(cls, cfg: DictConfig, params: Dict[str, str]) -> DictConfig:
        for k, v in params.items():
            cls.update(cfg, k, v)
        return cfg
