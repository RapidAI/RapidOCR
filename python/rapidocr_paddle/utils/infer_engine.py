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

from .logger import get_logger


class PaddleInferSession:
    def __init__(self, config, mode: Optional[str] = None) -> None:
        self.logger = get_logger("PaddleInferSession")
        self.mode = mode

        model_dir = Path(config["model_path"])
        pdmodel_path = model_dir / "inference.pdmodel"
        pdiparams_path = model_dir / "inference.pdiparams"

        self._verify_model(pdmodel_path)
        self._verify_model(pdiparams_path)

        infer_opts = inference.Config(str(pdmodel_path), str(pdiparams_path))

        if config["use_cuda"]:
            gpu_id = self.get_infer_gpuid()
            if gpu_id is None:
                self.logger.warning(
                    "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
                )
            infer_opts.enable_use_gpu(config["gpu_mem"], config["gpu_id"])
        else:
            infer_opts.disable_gpu()

        cpu_nums = os.cpu_count()
        infer_num_threads = config.get("cpu_math_library_num_threads", -1)
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
        return outputs

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")

    def get_input_tensors(
        self,
    ):
        input_names = self.predictor.get_input_names()
        for name in input_names:
            input_tensor = self.predictor.get_input_handle(name)
        return input_tensor

    def get_output_tensors(
        self,
    ):
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


class PaddleInferError(Exception):
    pass
