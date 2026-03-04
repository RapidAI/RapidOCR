# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import List

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
            if self.device.type=="mps":
                del inp
                torch.mps.empty_cache()
            return outputs

    def have_key(self, key: str = "character") -> bool:
        return False

    def get_character_list(self, key: str = "character") -> List[str]:
        return []


class TorchInferError(Exception):
    pass
