# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import abc
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
import requests
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ..utils.logger import Logger

cur_dir = Path(__file__).resolve().parent
MODEL_URL_PATH = cur_dir / "MODEL_URL.yaml"


logger = Logger(logger_name=__name__).get_log()


class Engine(Enum):
    ONNXRUNTIME = "onnxruntime"
    OPENVINO = "openvino"
    PADDLE = "paddlepaddle"
    TORCH = "torch"


def get_engine(engine_name: str):
    logger.info("Using engine_name: %s", engine_name)

    if engine_name == Engine.ONNXRUNTIME.value:
        from .onnxruntime import OrtInferSession

        return OrtInferSession

    if engine_name == Engine.OPENVINO.value:
        from .openvino import OpenVINOInferSession

        return OpenVINOInferSession

    if engine_name == Engine.PADDLE.value:
        from .paddlepaddle import PaddleInferSession

        return PaddleInferSession

    if engine_name == Engine.TORCH.value:
        from .torch import TorchInferSession

        return TorchInferSession

    raise ValueError(f"Unsupported engine: {engine_name}")


def get_engine_name(config: DictConfig) -> str:
    with_onnx = config.Global.with_onnx
    with_openvino = config.Global.with_openvino
    with_paddle = config.Global.with_paddle
    with_torch = config.Global.with_torch

    if with_onnx:
        return Engine.ONNXRUNTIME.value

    if with_openvino:
        return Engine.OPENVINO.value

    if with_paddle:
        return Engine.PADDLE.value

    if with_torch:
        return Engine.TORCH.value

    return Engine.ONNXRUNTIME.value


class InferSession(abc.ABC):
    model_info = OmegaConf.load(MODEL_URL_PATH)
    DEFAULT_MODE_PATH = cur_dir.parent / "models"
    logger = Logger(logger_name=__name__).get_log()

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

    @classmethod
    def download_file(cls, url: str, save_path: Union[str, Path]):
        cls.logger.info("Downloading model from %s to %s", url, save_path)
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
