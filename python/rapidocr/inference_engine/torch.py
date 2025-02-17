# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from ..networks.architectures.base_model import BaseModel
from ..utils.logger import Logger
from .base import InferSession

root_dir = Path(__file__).resolve().parent.parent
DEFAULT_CFG_PATH = root_dir / "networks" / "arch_config.yaml"


class TorchInferSession(InferSession):
    def __init__(self, config, mode: Optional[str] = None) -> None:
        self.logger = Logger(logger_name=__name__).get_log()

        all_arch_config = OmegaConf.load(DEFAULT_CFG_PATH)

        self.mode = mode
        model_path = Path(config["model_path"])
        self._verify_model(model_path)

        file_name = model_path.stem
        if file_name not in all_arch_config:
            raise ValueError(f"architecture {file_name} is not in arch_config.yaml")

        arch_config = all_arch_config[file_name]
        self.predictor = BaseModel(arch_config)
        self.predictor.load_state_dict(torch.load(model_path, weights_only=True))
        self.predictor.eval()

        self.use_gpu = False
        if config["use_cuda"]:
            self.predictor.cuda()
            self.use_gpu = True

    def __call__(self, img: np.ndarray):
        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu:
                inp = inp.cuda()

            # 适配跟onnx对齐取值逻辑
            outputs = self.predictor(inp).unsqueeze(0)
            return outputs.cpu().numpy()

    def have_key(self, key: str = "character") -> bool:
        return False


class TorchInferError(Exception):
    pass
