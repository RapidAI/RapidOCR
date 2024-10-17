# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import importlib
import sys
from pathlib import Path
from typing import Union

import requests
import yaml
from tqdm import tqdm


class BaseModule:
    def __init__(self, package_name: str = "rapidocr_onnxruntime"):
        self.package_name = package_name
        self.root_dir = Path(__file__).resolve().parent.parent
        self.package_dir = self.root_dir / self.package_name
        self.tests_dir = self.root_dir / "tests"

        sys.path.append(str(self.root_dir))
        sys.path.append(str(self.package_dir))

    def init_module(self, module_name: str, class_name: str = None):
        if class_name is None:
            module_part = importlib.import_module(f"{self.package_name}")
            return module_part
        module_part = importlib.import_module(f"{self.package_name}.{module_name}")
        return getattr(module_part, class_name)

    @staticmethod
    def read_yaml(yaml_path: str):
        with open(yaml_path, "rb") as f:
            data = yaml.load(f, Loader=yaml.Loader)
        return data


def download_file(url: str, save_path: Union[str, Path]):
    response = requests.get(url, stream=True, timeout=60)
    status_code = response.status_code

    if status_code != 200:
        raise DownloadModelError("Something went wrong while downloading models")

    total_size_in_bytes = int(response.headers.get("content-length", 1))
    block_size = 1024  # 1 Kibibyte
    with tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True) as pb:
        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                pb.update(len(data))
                file.write(data)


class DownloadModelError(Exception):
    pass
