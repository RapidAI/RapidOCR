# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import abc
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Union

import numpy as np
from omegaconf import OmegaConf

from ..utils.logger import Logger
from ..utils.typings import EngineType, ModelType, OCRVersion, TaskType
from ..utils.utils import import_package

cur_dir = Path(__file__).resolve().parent.parent
MODEL_URL_PATH = cur_dir / "default_models.yaml"

logger = Logger(logger_name=__name__).get_log()


def get_engine(engine_type: EngineType):
    logger.info("Using engine_name: %s", engine_type.value)

    if engine_type == EngineType.ONNXRUNTIME:
        if not import_package(engine_type.value):
            raise ImportError(f"{engine_type.value} is not installed.")

        from .onnxruntime import OrtInferSession

        return OrtInferSession

    if engine_type == EngineType.OPENVINO:
        if not import_package(engine_type.value):
            raise ImportError(f"{engine_type.value} is not installed")

        from .openvino import OpenVINOInferSession

        return OpenVINOInferSession

    if engine_type == EngineType.PADDLE:
        if not import_package(engine_type.value):
            raise ImportError(f"{engine_type.value} is not installed")

        from .paddle import PaddleInferSession

        return PaddleInferSession

    if engine_type == EngineType.TORCH:
        if not import_package(engine_type.value):
            raise ImportError(f"{engine_type.value} is not installed")

        from .torch import TorchInferSession

        return TorchInferSession

    raise ValueError(f"Unsupported engine: {engine_type.value}")


@dataclass
class FileInfo:
    engine_type: EngineType
    ocr_version: OCRVersion
    task_type: TaskType
    lang_type: Enum
    model_type: ModelType


class InferSession(abc.ABC):
    model_info = OmegaConf.load(MODEL_URL_PATH)
    DEFAULT_MODEL_PATH = cur_dir / "models"
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
    def get_model_url(cls, file_info: FileInfo) -> Dict[str, str]:
        model_dict = OmegaConf.select(
            cls.model_info,
            f"{file_info.engine_type.value}.{file_info.ocr_version.value}.{file_info.task_type.value}",
        )

        # 优先查找 server 模型
        if file_info.model_type == ModelType.SERVER:
            for k in model_dict:
                if (
                    k.startswith(file_info.lang_type.value)
                    and file_info.model_type.value in k
                ):
                    return model_dict[k]

        for k in model_dict:
            if k.startswith(file_info.lang_type.value):
                return model_dict[k]

        raise KeyError("File not found")

    @classmethod
    def get_dict_key_url(cls, file_info: FileInfo) -> str:
        model_dict = cls.get_model_url(file_info)
        return model_dict["dict_url"]
