# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import abc
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..utils.logger import Logger
from ..utils.utils import import_package

cur_dir = Path(__file__).resolve().parent.parent
MODEL_URL_PATH = cur_dir / "default_models.yaml"


logger = Logger(logger_name=__name__).get_log()


class Engine(Enum):
    ONNXRUNTIME = "onnxruntime"
    OPENVINO = "openvino"
    PADDLE = "paddlepaddle"
    TORCH = "torch"


def get_engine(engine_name: str):
    logger.info("Using engine_name: %s", engine_name)

    if engine_name == Engine.ONNXRUNTIME.value:
        if not import_package("onnxruntime"):
            raise ImportError("onnxruntime is not installed.")

        from .onnxruntime import OrtInferSession

        return OrtInferSession

    if engine_name == Engine.OPENVINO.value:
        if not import_package("openvino"):
            raise ImportError("openvino is not installed")

        from .openvino import OpenVINOInferSession

        return OpenVINOInferSession

    if engine_name == Engine.PADDLE.value:
        if not import_package("paddle"):
            raise ImportError("paddleopaddle is not installed")

        from .paddlepaddle import PaddleInferSession

        return PaddleInferSession

    if engine_name == Engine.TORCH.value:
        if not import_package("torch"):
            raise ImportError("torch is not installed")

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
    DEFAULT_MODE_PATH = cur_dir / "models"
    logger = Logger(logger_name=__name__).get_log()

    @abc.abstractmethod
    def __init__(self, config):
        pass

    @abc.abstractmethod
    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def _verify_model(model_path: Union[str, Path, None]):
        if model_path is None:
            raise ValueError("model_path is None!")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")

    @abc.abstractmethod
    def have_key(self, key: str = "character") -> bool:
        pass

    @classmethod
    def get_model_url(cls, engine_name: str, task_type: str, lang: str) -> str:
        lang, model_type = lang.split("_")
        model_dict = cls.model_info[engine_name]["PP-OCRv4"][task_type]

        # 优先查找 server 模型
        if model_type == "server":
            for k in model_dict:
                prefix = k.split("_")[0]
                if lang == prefix and "server" in k:
                    return model_dict[k]

        for k in model_dict:
            prefix = k.split("_")[0]
            if lang == prefix:
                return model_dict[k]

        raise KeyError("Model not found")
