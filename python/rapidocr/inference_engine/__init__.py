# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from enum import Enum

from omegaconf import DictConfig

from .onnxruntime import OrtInferSession
from .openvino import OpenVINOInferSession
from .paddlepaddle import PaddleInferSession
from .torch import TorchInferSession


class Engine(Enum):
    ONNXRUNTIME = "onnxruntime"
    OPENVINO = "openvino"
    PADDLE = "paddlepaddle"
    TORCH = "torch"


def get_engine(engine_name: str):
    if engine_name == Engine.ONNXRUNTIME.value:
        return OrtInferSession

    if engine_name == Engine.OPENVINO.value:
        return OpenVINOInferSession

    if engine_name == Engine.PADDLE.value:
        return PaddleInferSession

    if engine_name == Engine.TORCH.value:
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
