# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from ...utils.utils import mkdir
from ..base import InferSession
from .device_config import DeviceConfig
from .networks.main import ModelLoader


class TorchInferSession(InferSession):
    def __init__(self, cfg) -> None:
        self.model_root_dir = Path(cfg.get("model_root_dir"))
        mkdir(self.model_root_dir)

        self.device = DeviceConfig(cfg).setup_device()
        self.predictor = ModelLoader(cfg, self.device).predictor

    def __call__(self, img: np.ndarray):
        return self._run(img, reduce=False)

    def infer_argmax(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """For rec: argmax/max of the (B, T, vocab) logits on the device, so only
        (B, T) is copied to the CPU."""
        return self._run(img, reduce=True)

    def _run(self, img: np.ndarray, reduce: bool):
        with torch.no_grad():
            inp = torch.from_numpy(img)
            inp = inp.to(self.device)
            outputs = self.predictor(inp)
            if reduce:
                prob, idx = outputs.max(dim=2)
                outputs = (idx.cpu().numpy(), prob.cpu().numpy())
            else:
                outputs = outputs.cpu().numpy()
            if self.device.type == "mps":
                del inp
                torch.mps.empty_cache()
            return outputs

    def have_key(self, key: str = "character") -> bool:
        return False

    def get_character_list(self, key: str = "character") -> List[str]:
        return []


class TorchInferError(Exception):
    pass
