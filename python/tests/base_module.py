# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import importlib
import sys
from pathlib import Path

import yaml


class BaseModule():
    def __init__(self, package_name: str = 'rapidocr_onnxruntime'):
        self.package_name = package_name
        self.root_dir = Path(__file__).resolve().parent.parent
        self.package_dir = self.root_dir / self.package_name
        self.tests_dir = self.root_dir / 'tests'

        sys.path.append(str(self.root_dir))
        sys.path.append(str(self.package_dir))

    def init_module(self, module_name: str, class_name: str = None):
        if class_name is None:
            module_part = importlib.import_module(f'{self.package_name}')
            return module_part
        module_part = importlib.import_module(f'{self.package_name}.{module_name}')
        return getattr(module_part, class_name)

    @staticmethod
    def read_yaml(yaml_path: str):
        with open(yaml_path, 'rb') as f:
            data = yaml.load(f, Loader=yaml.Loader)
        return data
