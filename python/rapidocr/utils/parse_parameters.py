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
        self.lang_det_list = ["ch", "en", "multi"]
        self.lang_rec_list = [
            "ch",
            "en",
            "arabic",
            "chinese_cht",
            "cyrillic",
            "devanagari",
            "japan",
            "korean",
            "ka",
            "latin",
            "ta",
            "te",
        ]

    def __call__(self, lang_det: str, lang_rec: str) -> Tuple[str, str]:
        lang_det, det_model_type = lang_det.rsplit("_", 1)
        lang_det = self.parse_det_lang(lang_det)

        lang_rec, rec_model_type = lang_rec.rsplit("_", 1)
        lang_rec = self.parse_rec_lang(lang_rec)

        return f"{lang_det}_{det_model_type}", f"{lang_rec}_{rec_model_type}"

    def parse_det_lang(self, lang: str) -> str:
        lang = lang.strip().lower()

        if lang in self.lang_det_list:
            return lang
        raise ValueError(
            f"lang: {lang} is not in the supported list: {self.lang_det_list}"
        )

    def parse_rec_lang(self, lang: str) -> str:
        lang = lang.strip().lower()

        if lang in self.lang_rec_list:
            return lang

        raise ValueError(
            f"lang: {lang} is not in the supported list: {self.lang_rec_list}"
        )


class ParseParams(OmegaConf):
    def __init__(self):
        pass

    @classmethod
    def update_batch(cls, cfg: DictConfig, params: Dict[str, str]) -> DictConfig:
        global_keys = list(OmegaConf.to_container(cfg.Global).keys())
        for k, v in params.items():
            if k.startswith("Global") and k.split(".")[1] not in global_keys:
                raise ValueError(f"{k} is not a valid key.")
            cls.update(cfg, k, v)
        return cfg
