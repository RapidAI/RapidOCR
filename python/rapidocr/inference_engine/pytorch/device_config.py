# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

import torch

from ...utils.log import logger
from ...utils.typings import DeviceType
from ...utils.utils import mkdir

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent.parent
model_dir = root_dir / "models"


class DeviceConfig:
    def __init__(self, cfg):
        self.use_cuda = cfg.engine_cfg.use_cuda
        self.use_npu = cfg.engine_cfg.use_npu

        self.cfg = cfg

    def setup_device(self):
        if self.use_cuda:
            device_id = self.cfg.engine_cfg.cuda_ep_cfg.device_id
            return self.get_device(DeviceType.CUDA, device_id)

        if self.use_npu:
            device_id = self.cfg.engine_cfg.npu_ep_cfg.device_id
            return self.get_device(DeviceType.NPU, device_id)

        return self.get_device(DeviceType.CPU)

    def get_device(
        self, device_type: DeviceType = DeviceType.CPU, device_id: int = 0
    ) -> torch.device:
        if device_type == DeviceType.CPU:
            return self.config_cpu()

        if device_type == DeviceType.CUDA:
            return self.config_cuda(device_id)

        if device_type == DeviceType.NPU:
            return self.config_npu(device_id)

        raise DeviceConfigError(f"Unsupported device type: {device_type}")

    def config_cpu(self) -> torch.device:
        logger.info("Using CPU device")
        return torch.device("cpu")

    def config_cuda(self, device_id: int) -> torch.device:
        if not torch.cuda.is_available():
            raise DeviceConfigError("CUDA is not available.")

        logger.info(f"Using GPU device with ID: {device_id}")
        return torch.device(f"cuda:{device_id}")

    def config_npu(self, device_id: int) -> torch.device:
        try:
            import torch_npu
        except ImportError as e:
            raise ImportError(
                "torch_npu is not installed. \n"
                "Please refer to https://github.com/Ascend/pytorch to see how to install."
            )

        if not torch_npu.npu.is_available():
            raise DeviceConfigError("NPU is not available.")

        kernel_meta_dir = (model_dir / "kernel_meta").resolve()
        mkdir(kernel_meta_dir)

        options = {
            "ACL_OP_COMPILER_CACHE_MODE": "enable",
            "ACL_OP_COMPILER_CACHE_DIR": str(kernel_meta_dir),
        }
        torch_npu.npu.set_option(options)

        logger.info(f"Using NPU device with ID: {device_id}")
        return torch.device(f"npu:{device_id}")


class DeviceConfigError(Exception):
    pass
