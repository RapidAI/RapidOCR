# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ..networks.architectures.base_model import BaseModel
from ..utils.download_file import DownloadFile, DownloadFileInput
from ..utils.log import logger
from .base import FileInfo, InferSession

root_dir = Path(__file__).resolve().parent.parent
DEFAULT_CFG_PATH = root_dir / "networks" / "arch_config.yaml"


class TorchInferSession(InferSession):
    def __init__(self, cfg) -> None:
        self.logger = logger

        model_path = cfg.get("model_path", None)
        if model_path is None:
            model_info = self.get_model_url(
                FileInfo(
                    engine_type=cfg.engine_type,
                    ocr_version=cfg.ocr_version,
                    task_type=cfg.task_type,
                    lang_type=cfg.lang_type,
                    model_type=cfg.model_type,
                )
            )
            default_model_url = model_info["model_dir"]
            model_path = self.DEFAULT_MODEL_PATH / Path(default_model_url).name
            DownloadFile.run(
                DownloadFileInput(
                    file_url=default_model_url,
                    sha256=model_info["SHA256"],
                    save_path=model_path,
                    logger=self.logger,
                )
            )

        self.logger.info(f"Using {model_path}")
        model_path = Path(model_path)
        self._verify_model(model_path)

        all_arch_config = OmegaConf.load(DEFAULT_CFG_PATH)
        file_name = model_path.stem
        if file_name not in all_arch_config:
            raise ValueError(f"architecture {file_name} is not in arch_config.yaml")

        arch_config = all_arch_config.get(file_name)
        self.predictor = BaseModel(arch_config)
        self.predictor.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
        self.predictor.eval()
        self.use_gpu = False
        self.use_npu = False
        if cfg.engine_cfg.use_cuda:
            self.device = torch.device(f"cuda:{cfg.engine_cfg.gpu_id}")
            self.predictor.to(self.device)
            self.use_gpu = True
        elif cfg.engine_cfg.use_npu:
            try:
                import torch_npu
                options = {
                    # 设定算子编译的磁盘缓存模式，非必要每次重新编译
                    "ACL_OP_COMPILER_CACHE_MODE": "enable",
                    # 指定缓存目录，确保路径已存在
                    "ACL_OP_COMPILER_CACHE_DIR": "./kernel_meta",
                }
                torch_npu.npu.set_option(options)
            except ImportError:
                self.logger.warning("torch_npu is not installed, options with ACL setting failed.")
            self.device = torch.device(f"npu:{cfg.engine_cfg.npu_id}")
            self.predictor.to(self.device)
            self.use_npu = True

    def __call__(self, img: np.ndarray):
        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu or self.use_npu:
                inp = inp.to(self.device)

            # 适配跟onnx对齐取值逻辑
            outputs = self.predictor(inp).cpu().numpy()
            return outputs

    def have_key(self, key: str = "character") -> bool:
        return False


class TorchInferError(Exception):
    pass
