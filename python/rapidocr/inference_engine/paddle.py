# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import platform
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import paddle
from omegaconf.errors import ConfigKeyError
from paddle import inference

from ..utils.download_file import DownloadFile, DownloadFileInput
from ..utils.logger import Logger
from ..utils.typings import OCRVersion
from .base import FileInfo, InferSession


class PaddleInferSession(InferSession):
    def __init__(self, cfg, mode: Optional[str] = None) -> None:
        self.logger = Logger(logger_name=__name__).get_log()
        self.mode = mode

        pdmodel_path, pdiparams_path = self._setup_model(cfg)
        if cfg.ocr_version == OCRVersion.PPOCRV5:
            self._init_predictor_v2(cfg, pdmodel_path, pdiparams_path)
        else:
            self._init_predictor_v1(cfg, pdmodel_path, pdiparams_path)

    def _setup_model(self, cfg) -> Tuple[Path, Path]:
        pdmodel_name = "inference.json"
        pdmodel_name_v2 = "inference.pdmodel"
        pdiparams_name = "inference.pdiparams"

        model_dir = cfg.get("model_dir", None)
        if model_dir is None:
            model_info = self.get_model_url(
                FileInfo(
                    engine_type=cfg.engine_type,
                    ocr_version=cfg.ocr_version,
                    task_type=cfg.task_type,
                    lang_type=cfg.lang_type,
                    model_type=cfg.model_type,
                )
            )
            default_model_dir = model_info["model_dir"]

            try:
                pdmodel_path = self.download_model(
                    model_info, default_model_dir, pdmodel_name
                )
            except ConfigKeyError as e:
                pdmodel_path = self.download_model(
                    model_info, default_model_dir, pdmodel_name_v2
                )
            except Exception as e:
                raise PaddleInferError(f"Download model error: {e}") from e

            pdiparams_path = self.download_model(
                model_info, default_model_dir, pdiparams_name
            )

            self.logger.info(f"Using {pdmodel_path}")
            self.logger.info(f"Using {pdiparams_path}")
            return pdmodel_path, pdiparams_path

        model_dir = Path(model_dir)
        pdmodel_path = model_dir / pdmodel_name
        pdiparams_path = model_dir / pdiparams_name
        if not pdmodel_path.exists():
            pdmodel_path = model_dir / pdmodel_name_v2
        self._verify_model(pdmodel_path)
        self._verify_model(pdiparams_path)

        self.logger.info(f"Using {pdmodel_path}")
        self.logger.info(f"Using {pdiparams_path}")
        return pdmodel_path, pdiparams_path

    def download_model(
        self, model_info, default_model_dir: str, model_file_name: str
    ) -> Path:
        model_file_url = f"{default_model_dir}/{model_file_name}"
        model_file_path = (
            self.DEFAULT_MODEL_PATH / Path(default_model_dir).name / model_file_name
        )
        DownloadFile.run(
            DownloadFileInput(
                file_url=model_file_url,
                sha256=model_info[model_file_name],
                save_path=model_file_path,
                logger=self.logger,
            )
        )
        return model_file_path

    def _init_predictor_v1(self, cfg, pdmodel_path, pdiparams_path):
        infer_opts = inference.Config(str(pdmodel_path), str(pdiparams_path))

        if cfg.engine_cfg.use_cuda:
            gpu_id = self.get_infer_gpuid()
            if gpu_id is None:
                self.logger.warning(
                    "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
                )
            infer_opts.enable_use_gpu(cfg.engine_cfg.gpu_mem, cfg.engine_cfg.gpu_id)
        else:
            infer_opts.disable_gpu()

        cpu_nums = os.cpu_count()
        infer_num_threads = cfg.engine_cfg.get("cpu_math_library_num_threads", -1)
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

    def _init_predictor_v2(self, cfg, pdmodel_path, pdiparams_path):
        infer_opts = inference.Config(str(pdmodel_path), str(pdiparams_path))

        if cfg.engine_cfg.use_cuda:
            gpu_id = self.get_infer_gpuid()
            if gpu_id is None:
                self.logger.warning(
                    "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
                )
            infer_opts.enable_use_gpu(cfg.engine_cfg.gpu_mem, cfg.engine_cfg.gpu_id)
        else:
            infer_opts.disable_gpu()
            if hasattr(infer_opts, "disable_mkldnn"):
                infer_opts.disable_mkldnn()

            cpu_nums = os.cpu_count()
            infer_num_threads = cfg.engine_cfg.get("cpu_math_library_num_threads", -1)
            if infer_num_threads != -1 and 1 <= infer_num_threads <= cpu_nums:
                infer_opts.set_cpu_math_library_num_threads(infer_num_threads)

            if hasattr(infer_opts, "enable_new_ir"):
                infer_opts.enable_new_ir(True)

            if hasattr(infer_opts, "enable_new_executor"):
                infer_opts.enable_new_executor()
            infer_opts.set_optimization_level(3)

        infer_opts.enable_memory_optim()
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
