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
from ..utils.utils import mkdir
from .base import FileInfo, InferSession

root_dir = Path(__file__).resolve().parent.parent
DEFAULT_CFG_PATH = root_dir / "networks" / "arch_config.yaml"


class TorchInferSession(InferSession):
    def __init__(self, cfg) -> None:
        model_path = self._init_model_path(cfg)
        arch_config = self._load_arch_config(model_path)

        self.predictor = self._build_and_load_model(arch_config, model_path)

        self._setup_device(cfg)

        self.predictor.eval()

    def _init_model_path(self, cfg) -> Path:
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
                    logger=logger,
                )
            )

        logger.info(f"Using {model_path}")
        self._verify_model(model_path)
        return Path(model_path)

    def _load_arch_config(self, model_path: Path):
        all_arch_config = OmegaConf.load(DEFAULT_CFG_PATH)

        file_name = model_path.stem
        if file_name not in all_arch_config:
            raise ValueError(f"architecture {file_name} is not in arch_config.yaml")

        return all_arch_config.get(file_name)

    def _build_and_load_model(self, arch_config, model_path: Path):
        model = BaseModel(arch_config)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        return model

    def _setup_device(self, cfg):
        self.device, self.use_gpu, self.use_npu = self._resolve_device_config(cfg)

        if self.use_npu:
            self._config_npu()

        self._move_model_to_device()

    def _resolve_device_config(self, cfg):
        if cfg.engine_cfg.use_cuda:
            return torch.device(f"cuda:{cfg.engine_cfg.gpu_id}"), True, False

        if cfg.engine_cfg.use_npu:
            return torch.device(f"npu:{cfg.engine_cfg.npu_id}"), False, True

        return torch.device("cpu"), False, False

    def _config_npu(self):
        try:
            import torch_npu

            kernel_meta_dir = (root_dir / "kernel_meta").resolve()
            mkdir(kernel_meta_dir)

            options = {
                "ACL_OP_COMPILER_CACHE_MODE": "enable",
                "ACL_OP_COMPILER_CACHE_DIR": str(kernel_meta_dir),
            }
            torch_npu.npu.set_option(options)
        except ImportError:
            logger.warning(
                "torch_npu is not installed, options with ACL setting failed. \n"
                "Please refer to https://github.com/Ascend/pytorch to see how to install."
            )

            self.device = torch.device("cpu")
            self.use_npu = False

    def _move_model_to_device(self):
        self.predictor.to(self.device)

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
