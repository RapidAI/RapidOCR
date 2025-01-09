# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import platform
from pathlib import Path
from typing import Optional, Union, Dict

import numpy as np
import torch
import yaml

root_dir = Path(__file__).resolve().parent.parent
DEFAULT_CFG_PATH = root_dir / "arch_config.yaml"

def read_yaml(yaml_path: Union[str, Path]) -> Dict[str, Dict]:
    with open(yaml_path, "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data

from .logger import get_logger
from rapidocr_torch.modeling.architectures.base_model import BaseModel


class TorchInferSession:
    def __init__(self, config, mode: Optional[str] = None) -> None:

        all_arch_config = read_yaml(DEFAULT_CFG_PATH)

        self.logger = get_logger("TorchInferSession")
        self.mode = mode
        model_path = Path(config["model_path"])
        self._verify_model(model_path)
        file_name = model_path.stem
        if file_name not in all_arch_config:
            raise ValueError(f"architecture {file_name} is not in config.yaml")
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
    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


class TorchInferError(Exception):
    pass
