# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Dict, Union

import yaml

from .infer_engine import OrtInferSession
from .load_image import LoadImage, LoadImageError
from .logger import get_logger
from .parse_parameters import UpdateParameters, init_args, update_model_path
from .vis_res import VisRes


def read_yaml(yaml_path: Union[str, Path]) -> Dict[str, Dict]:
    with open(yaml_path, "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data
