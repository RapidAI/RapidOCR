# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com

import importlib
from pathlib import Path
import sys

import yaml

package_name = 'rapidocr_onnxruntime'
# package_name = 'rapidocr_openvino'

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / package_name))


def init_module(module_name, class_name):
    module_part = importlib.import_module(module_name)
    return getattr(module_part, class_name)


def read_yaml(yaml_path):
    with open(yaml_path, 'rb') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data
