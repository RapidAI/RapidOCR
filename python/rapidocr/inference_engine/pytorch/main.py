# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import numpy as np
import torch

from ..base import InferSession
from .device_config import DeviceConfig
from .networks.main import ModelLoader


class TorchInferSession(InferSession):
    def __init__(self, cfg) -> None:
        self.device = DeviceConfig(cfg).setup_device()
        self.predictor = ModelLoader(cfg, self.device).predictor

    def __call__(self, img: np.ndarray):
        with torch.no_grad():
            inp = torch.from_numpy(img)
            inp = inp.to(self.device)
            outputs = self.predictor(inp).cpu().numpy()
            return outputs

    def have_key(self, key: str = "character") -> bool:
        return False


class TorchInferError(Exception):
    pass
