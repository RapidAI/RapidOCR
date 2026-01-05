# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import platform

import paddle

from ...utils.log import logger
from ...utils.typings import OCRVersion


class DeviceConfig:
    def __init__(self, cfg, infer_opts, ocr_version):
        self.use_cuda = cfg.engine_cfg.use_cuda
        self.use_npu = cfg.engine_cfg.use_npu

        self.cfg = cfg
        self.infer_opts = infer_opts
        self.ocr_version = ocr_version

    def setup_device(self):
        if self.use_cuda:
            self.config_cuda(self.cfg.engine_cfg.cuda_ep_cfg)
            return self.infer_opts

        if self.use_npu:
            self.config_npu(self.cfg.engine_cfg.npu_ep_cfg)
            return self.infer_opts

        self.config_cpu()
        return self.infer_opts

    def config_cuda(self, ep_cfg):
        if not self.check_cuda():
            raise DeviceConfigError("CUDA is not available.")

        gpu_id = self.get_infer_gpuid()
        if gpu_id is None:
            raise DeviceConfigError(
                "CUDA is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
            )

        self.infer_opts.enable_use_gpu(ep_cfg.gpu_mem, ep_cfg.device_id)
        logger.info(f"Using CUDA device with ID: {ep_cfg.device_id}")

    def config_npu(self, ep_cfg):
        self.setup_device_envs(ep_cfg.envs)

        npu_id = ep_cfg.device_id
        self.infer_opts.enable_custom_device("npu", npu_id)
        logger.info(f"Using NPU device with ID: {npu_id}")

    def config_cpu(self):
        self.infer_opts.disable_gpu()
        if hasattr(self.infer_opts, "disable_mkldnn"):
            self.infer_opts.disable_mkldnn()
        logger.info("Using CPU device")

        cpu_nums = os.cpu_count()
        infer_num_threads = self.cfg.get("cpu_math_library_num_threads", -1)
        if infer_num_threads != -1 and 1 <= infer_num_threads <= cpu_nums:
            self.infer_opts.set_cpu_math_library_num_threads(infer_num_threads)

            logger.info(f"Set CPU math library threads to: {infer_num_threads}")

        if self.ocr_version == OCRVersion.PPOCRV5:
            if hasattr(self.infer_opts, "enable_new_ir"):
                self.infer_opts.enable_new_ir(True)

            if hasattr(self.infer_opts, "enable_new_executor"):
                self.infer_opts.enable_new_executor()
            self.infer_opts.set_optimization_level(3)

    @staticmethod
    def setup_device_envs(envs):
        for key, val in envs.items():
            os.environ[key] = str(val)
            logger.info(f"{key} has been set to {val}.")

    @staticmethod
    def check_cuda() -> bool:
        if paddle.is_compiled_with_cuda():
            device = paddle.get_device()
            if device.startswith("gpu"):
                logger.info(f"GPU is available. Current device: {device}")
                return True

            logger.warning(
                "PaddlePaddle was compiled with CUDA support, but no GPU is currently available."
            )
            return False

        logger.warning(
            "PaddlePaddle was not compiled with CUDA support (CPU-only version)."
        )
        return False

    @staticmethod
    def get_infer_gpuid() -> int:
        sysstr = platform.system()
        if sysstr == "Windows":
            return 0

        if not paddle.device.is_compiled_with_rocm:
            cmd = "env | grep CUDA_VISIBLE_DEVICES"
        else:
            cmd = "env | grep HIP_VISIBLE_DEVICES"
        env_cuda = os.popen(cmd).readlines()

        if len(env_cuda) == 0:
            return 0

        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])


class DeviceConfigError(Exception):
    pass
