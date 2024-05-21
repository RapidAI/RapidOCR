# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import platform
import traceback
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers,
    get_device,
)

from .logger import get_logger


class EP(Enum):
    CPU_EP = "CPUExecutionProvider"
    CUDA_EP = "CUDAExecutionProvider"
    DIRECTML_EP = "DmlExecutionProvider"


class OrtInferSession:
    def __init__(self, config):
        self.logger = get_logger("OrtInferSession")

        model_path = config.get("model_path", None)
        self._verify_model(model_path)

        self.cfg_use_cuda = config.get("use_cuda", None)
        self.cfg_use_dml = config.get("use_dml", None)

        self.had_providers: List[str] = get_available_providers()
        EP_list = self._get_ep_list()

        sess_opt = self._init_sess_opts(config)
        self.session = InferenceSession(
            model_path, sess_options=sess_opt, providers=[item[0].value for item in EP_list]
        )
        self._verify_providers()

    @staticmethod
    def _init_sess_opts(config):
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cpu_nums = os.cpu_count()
        intra_op_num_threads = config.get("intra_op_num_threads", -1)
        if intra_op_num_threads != -1 and 1 <= intra_op_num_threads <= cpu_nums:
            sess_opt.intra_op_num_threads = intra_op_num_threads

        inter_op_num_threads = config.get("inter_op_num_threads", -1)
        if inter_op_num_threads != -1 and 1 <= inter_op_num_threads <= cpu_nums:
            sess_opt.inter_op_num_threads = inter_op_num_threads

        return sess_opt

    def _get_ep_list(self) -> List[Tuple[str, str]]:
        cpu_provider_opts = {
            "arena_extend_strategy": "kSameAsRequested",
        }
        EP_list = [(EP.CPU_EP, cpu_provider_opts)]

        cuda_provider_opts = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }
        self.use_cuda = self._check_cuda()
        if self.use_cuda:
            EP_list.insert(0, (EP.CUDA_EP, cuda_provider_opts))

        self.use_directml = self._check_dml()
        if self.use_directml:
            self.logger.info(
                "Windows 10 or above detected, try to use DirectML as primary provider"
            )
            directml_options = (
                cuda_provider_opts if self.use_cuda else cpu_provider_opts
            )
            EP_list.insert(0, (EP.DIRECTML_EP, directml_options))
        return EP_list

    def _check_cuda(self) -> bool:
        if not self.cfg_use_cuda:
            return False

        cur_device = get_device()
        if cur_device == "GPU" and EP.CUDA_EP.value in self.had_providers:
            return True

        self.logger.warning(
            "%s is not in available providers (%s). Use %s inference by default.",
            EP.CUDA_EP.value,
            self.had_providers,
            self.had_providers[0],
        )
        self.logger.info("!!!Recommend to use rapidocr_paddle for inference on GPU.")
        self.logger.info(
            "(For reference only) If you want to use GPU acceleration, you must do:"
        )
        self.logger.info(
            "First, uninstall all onnxruntime pakcages in current environment."
        )
        self.logger.info(
            "Second, install onnxruntime-gpu by `pip install onnxruntime-gpu`."
        )
        self.logger.info(
            "\tNote the onnxruntime-gpu version must match your cuda and cudnn version."
        )
        self.logger.info(
            "\tYou can refer this link: https://onnxruntime.ai/docs/execution-providers/CUDA-EP.html"
        )
        self.logger.info(
            "Third, ensure %s is in available providers list. e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']",
            EP.CUDA_EP.value,
        )
        return False

    def _check_dml(self) -> bool:
        if not self.cfg_use_dml:
            return False

        cur_os = platform.system()
        if cur_os != "Windows":
            self.logger.warning(
                "DirectML is only supported in Windows OS. The current OS is %s. Use %s inference by default.",
                cur_os,
                self.had_providers[0],
            )
            return False

        cur_window_version = int(platform.release().split(".")[0])
        if cur_window_version < 10:
            self.logger.warning(
                "DirectML is only supported in Windows 10 and above OS. The current Windows version is %s. Use %s inference by default.",
                cur_window_version,
                self.had_providers[0],
            )
            return False

        if EP.DIRECTML_EP.value in self.had_providers:
            return True

        self.logger.warning(
            "%s is not in available providers (%s). Use %s inference by default.",
            EP.DIRECTML_EP.value,
            self.had_providers,
            self.had_providers[0],
        )
        self.logger.info("If you want to use DirectML acceleration, you must do:")
        self.logger.info(
            "First, uninstall all onnxruntime pakcages in current environment."
        )
        self.logger.info(
            "Second, install onnxruntime-directml by `pip install onnxruntime-directml`"
        )
        self.logger.info(
            "Third, ensure %s is in available providers list. e.g. ['DmlExecutionProvider', 'CPUExecutionProvider']",
            EP.DIRECTML_EP.value,
        )
        return False

    def _verify_providers(self) -> None:
        session_providers = self.session.get_providers()
        first_provider = session_providers[0]

        if self.use_cuda and first_provider != EP.CUDA_EP.value:
            self.logger.warning(
                "%s is not avaiable for current env, the inference part is automatically shifted to be executed under %s.",
                EP.CUDA_EP.value,
                first_provider,
            )

        if self.use_directml and first_provider != EP.DIRECTML_EP.value:
            self.logger.warning(
                "%s is not available for current env, the inference part is automatically shifted to be executed under %s.",
                EP.DIRECTML_EP.value,
                first_provider,
            )

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), [input_content]))
        try:
            return self.session.run(self.get_output_names(), input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(self):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path: Union[str, Path, None]):
        if model_path is None:
            raise ValueError("model_path is None!")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


class ONNXRuntimeError(Exception):
    pass
