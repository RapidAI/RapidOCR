# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
from paddle import inference

from ..utils.logger import Logger
from .base import InferSession


class PaddleInferSession(InferSession):
    def __init__(self, config, mode: Optional[str] = None) -> None:
        self.logger = Logger(logger_name=__name__).get_log()
        self.mode = mode

        model_dir = Path(config.model_dir)
        pdmodel_path = model_dir / "inference.pdmodel"
        pdiparams_path = model_dir / "inference.pdiparams"

        if not self._verify_model(pdmodel_path):
            self.logger.warning(
                "%s model path is invalid, try to download the default model.",
                pdmodel_path,
            )
            default_model_url = self.get_model_url(
                config.engine_name, config.task_type, config.lang
            )
            default_model_url = f"{default_model_url}/{pdmodel_path.name}"
            model_path = model_dir / pdmodel_path.name
            self.download_file(default_model_url, model_path)

        if not self._verify_model(pdiparams_path):
            self.logger.warning(
                "%s model path is invalid, try to download the default model.",
                pdiparams_path,
            )
            default_model_url = self.get_model_url(
                config.engine_name, config.task_type, config.lang
            )
            default_model_url = f"{default_model_url}/{pdiparams_path.name}"
            model_path = model_dir / pdiparams_path.name
            self.download_file(default_model_url, model_path)

        infer_opts = inference.Config(str(pdmodel_path), str(pdiparams_path))

        if config.engine_cfg.use_cuda:
            gpu_id = self.get_infer_gpuid()
            if gpu_id is None:
                self.logger.warning(
                    "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
                )
            infer_opts.enable_use_gpu(
                config.engine_cfg.gpu_mem, config.engine_cfg.gpu_id
            )
        else:
            infer_opts.disable_gpu()

        cpu_nums = os.cpu_count()
        infer_num_threads = config.engine_cfg.get("cpu_math_library_num_threads", -1)
        if infer_num_threads != -1 and 1 <= infer_num_threads <= cpu_nums:
            infer_opts.set_cpu_math_library_num_threads(infer_num_threads)

        # enable memory optim
        infer_opts.enable_memory_optim()
        infer_opts.disable_glog_info()
        infer_opts.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        infer_opts.delete_pass("matmul_transpose_reshape_fuse_pass")
        infer_opts.switch_use_feed_fetch_ops(False)
        infer_opts.switch_ir_optim(True)

        self.predictor = inference.create_predictor(infer_opts)

    def __call__(self, img: np.ndarray):
        input_tensor = self.get_input_tensors()
        output_tensors = self.get_output_tensors()

        input_tensor.copy_from_cpu(img)
        self.predictor.run()

        outputs = []
        for output_tensor in output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        self.predictor.try_shrink_memory()
        return outputs[0]

    def get_input_tensors(self):
        input_names = self.predictor.get_input_names()
        for name in input_names:
            input_tensor = self.predictor.get_input_handle(name)
        return input_tensor

    def get_output_tensors(self):
        output_names = self.predictor.get_output_names()
        if self.mode == "rec":
            output_name = "softmax_0.tmp_0"
            if output_name in output_names:
                return [self.predictor.get_output_handle(output_name)]

        output_tensors = []
        for output_name in output_names:
            output_tensor = self.predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
        return output_tensors

    @staticmethod
    def get_infer_gpuid():
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

    def have_key(self, key: str = "character") -> bool:
        return False


class PaddleInferError(Exception):
    pass
