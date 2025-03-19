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
from ..utils.utils import download_file
from .base import InferSession

root_dir = Path(__file__).resolve().parent.parent
DEFAULT_CFG_PATH = root_dir / "networks" / "arch_config.yaml"


class TorchInferSession(InferSession):
    def __init__(self, config, mode: Optional[str] = None) -> None:
        self.logger = Logger(logger_name=__name__).get_log()
        self.mode = mode

        model_path = config.get("model_path", None)
        if model_path is None:
            default_model_url = self.get_model_url(
                config.engine_name, config.task_type, config.lang
            )
            if self.mode == "rec":
                default_model_url = default_model_url["model_dir"]

            model_path = self.DEFAULT_MODE_PATH / Path(default_model_url).name
            download_file(default_model_url, model_path, self.logger)

        self._verify_model(model_path)

        all_arch_config = OmegaConf.load(DEFAULT_CFG_PATH)
        file_name = model_path.stem
        if file_name not in all_arch_config:
            raise ValueError(f"architecture {file_name} is not in arch_config.yaml")

        arch_config = all_arch_config[file_name]
        self.predictor = BaseModel(arch_config)
        self.predictor.load_state_dict(torch.load(model_path, weights_only=True))
        self.predictor.eval()

        self.use_gpu = False
        if config.engine_cfg.use_cuda:
            self.device = torch.device(f"cuda:{config.engine_cfg.gpu_id}")
            self.predictor.to(self.device)
            self.use_gpu = True

    def __call__(self, img: np.ndarray):
        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu:
                inp = inp.to(self.device)

            # 适配跟onnx对齐取值逻辑
            outputs = self.predictor(inp).cpu().numpy()
            return outputs

    def have_key(self, key: str = "character") -> bool:
        return False


class TorchInferError(Exception):
    pass
