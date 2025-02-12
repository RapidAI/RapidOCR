# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import abc
from pathlib import Path
from typing import Union

import numpy as np
import requests
from omegaconf import OmegaConf
from tqdm import tqdm

cur_dir = Path(__file__).resolve().parent
MODEL_URL_PATH = cur_dir / "MODEL_URL.yaml"


class InferSession(abc.ABC):
    model_info = OmegaConf.load(MODEL_URL_PATH)
    DEFAULT_MODE_PATH = cur_dir.parent / "models"

    @abc.abstractmethod
    def __init__(self, config):
        pass

    @abc.abstractmethod
    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def _verify_model(model_path: Union[str, Path, None]) -> bool:
        if model_path is None:
            return False

        model_path = Path(model_path)
        if not model_path.exists():
            return False

        if not model_path.is_file():
            return False

        return True

    @abc.abstractmethod
    def have_key(self, key: str = "character") -> bool:
        pass

    @staticmethod
    def download_file(url: str, save_path: Union[str, Path]):
        print(f"Downloading model from {url} to {save_path}")
        response = requests.get(url, stream=True, timeout=60)
        status_code = response.status_code

        if status_code != 200:
            raise DownloadFileError("Something went wrong while downloading models")

        total_size_in_bytes = int(response.headers.get("content-length", 1))
        block_size = 1024  # 1 Kibibyte
        with tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True) as pb:
            with open(save_path, "wb") as file:
                for data in response.iter_content(block_size):
                    pb.update(len(data))
                    file.write(data)

    @classmethod
    def get_model_url(cls, engine_name: str, task_type: str, lang: str) -> str:
        model_dict = cls.model_info[engine_name]["PP-OCRv4"][task_type]
        for k in model_dict:
            prefix = k.split("_")[0]
            if lang == prefix:
                return model_dict[k]
        raise KeyError("Model not found")


class DownloadFileError(Exception):
    pass
