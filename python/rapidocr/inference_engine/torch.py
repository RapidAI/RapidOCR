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

        self.device = self._setup_device(cfg)

        self.predictor.to(self.device)
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

    def _setup_device(self, cfg) -> torch.device:
        self.use_cuda = cfg.engine_cfg.use_cuda
        self.use_npu = cfg.engine_cfg.use_npu

        if self.use_cuda:
            return self._config_cuda(cfg.engine_cfg.cuda_ep_cfg.device_id)

        if self.use_npu:
            return self._config_npu(cfg.engine_cfg.npu_ep_cfg.device_id)

        return self._config_cpu()

    def _config_cpu(self) -> torch.device:
        logger.info("Using CPU device")
        return torch.device("cpu")

    def _config_cuda(self, device_id: int) -> torch.device:
        if not torch.cuda.is_available():
            raise TorchInferError("CUDA is not available.")

        logger.info(f"Using GPU device with ID: {device_id}")
        return torch.device(f"cuda:{device_id}")

    def _config_npu(self, device_id: int) -> torch.device:
        try:
            import torch_npu
        except ImportError as e:
            logger.warning(
                "torch_npu is not installed. \n"
                "Please refer to https://github.com/Ascend/pytorch to see how to install."
            )
            self.use_npu = False
            logger.warning("Roll back to CPU device")
            return torch.device("cpu")

        try:
            if not torch_npu.npu.is_available():
                raise TorchInferError("NPU is not available.")
        except TorchInferError as e:
            logger.warning(e)
            self.use_npu = False
            logger.warning("Roll back to CPU device")
            return torch.device("cpu")

        kernel_meta_dir = (root_dir / "kernel_meta").resolve()
        mkdir(kernel_meta_dir)

        options = {
            "ACL_OP_COMPILER_CACHE_MODE": "enable",
            "ACL_OP_COMPILER_CACHE_DIR": str(kernel_meta_dir),
        }
        torch_npu.npu.set_option(options)

        logger.info(f"Using NPU device with ID: {device_id}")
        return torch.device(f"npu:{device_id}")

    def __call__(self, img: np.ndarray):
        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_cuda or self.use_npu:
                inp = inp.to(self.device)

            # 适配onnx对齐取值逻辑
            outputs = self.predictor(inp).cpu().numpy()
            return outputs

    def have_key(self, key: str = "character") -> bool:
        return False


class TorchInferError(Exception):
    pass
