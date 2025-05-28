# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Dict, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
InputType = Union[str, np.ndarray, bytes, Path, Image.Image]


class ParseParams(OmegaConf):
    def __init__(self):
        pass

    @staticmethod
    def load(file: Union[str, Path]):
        cfg = OmegaConf.load(file)
        return cfg

    @classmethod
    def update_batch(cls, cfg: DictConfig, params: Dict[str, str]) -> DictConfig:
        global_keys = list(OmegaConf.to_container(cfg.Global).keys())
        for k, v in params.items():
            if k.startswith("Global") and k.split(".")[1] not in global_keys:
                raise ValueError(f"{k} is not a valid key.")
            cls.update(cfg, k, v)
        return cfg
