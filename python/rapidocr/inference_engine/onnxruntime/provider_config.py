# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import platform
from enum import Enum
from typing import Any, Dict, List, Sequence

from onnxruntime import get_available_providers, get_device

from ...utils.logger import Logger


class EP(Enum):
    CPU_EP = "CPUExecutionProvider"
    CUDA_EP = "CUDAExecutionProvider"
    DIRECTML_EP = "DmlExecutionProvider"
    CANN_EP = "CANNExecutionProvider"


class ProviderConfig:
    def __init__(self, cfg_use_cuda: bool, cfg_use_dml: bool, cfg_use_cann: bool):
        self.logger = Logger(logger_name=__name__).get_log()

        self.had_providers: List[str] = get_available_providers()
        self.default_provider = self.had_providers[0]

        self.cfg_use_cuda = cfg_use_cuda
        self.cfg_use_dml = cfg_use_dml
        self.cfg_use_cann = cfg_use_cann

    def get_ep_list(self):
        results = [(EP.CPU_EP.value, self.cpu_ep_cfg())]

        if self.is_cuda_available():
            results.insert(0, (EP.CUDA_EP.value, self.cuda_ep_cfg()))

        if self.is_dml_available():
            self.logger.info(
                "Windows 10 or above detected, try to use DirectML as primary provider"
            )
            results.insert(0, (EP.DIRECTML_EP.value, self.dml_ep_cfg()))

        if self.is_cann_available():
            self.logger.info("Try to use CANNExecutionProvider to infer")
            results.insert(0, (EP.CANN_EP.value, self.cann_ep_cfg()))

        return results

    def cpu_ep_cfg(self) -> Dict[str, Any]:
        return {
            "arena_extend_strategy": "kSameAsRequested",
        }

    def cuda_ep_cfg(self) -> Dict[str, Any]:
        return {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }

    def dml_ep_cfg(self) -> Dict[str, Any]:
        if self.use_cuda:
            return self.cuda_ep_cfg()
        return self.cpu_ep_cfg()

    def cann_ep_cfg(self) -> Dict[str, Any]:
        return {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "npu_mem_limit": 20 * 1024 * 1024 * 1024,
            "op_select_impl_mode": "high_performance",
            "optypelist_for_implmode": "Gelu",
            "enable_cann_graph": True,
        }

    def verify_providers(self, session_providers: Sequence[str]):
        if not session_providers:
            raise ValueError("Session Providers is empty")

        first_provider = session_providers[0]

        providers_to_check = {
            EP.CUDA_EP: self.is_cuda_available,
            EP.DIRECTML_EP: self.is_dml_available,
            EP.CANN_EP: self.is_cann_available,
        }

        for ep, check_func in providers_to_check.items():
            if check_func() and first_provider != ep.value:
                self.logger.warning(
                    f"{ep.value} is available, but the inference part is automatically shifted to be executed under {first_provider}. "
                )
                self.logger.warning(f"The available lists are {session_providers}")

    def is_cuda_available(self) -> bool:
        if not self.cfg_use_cuda:
            return False

        CUDA_EP = EP.CUDA_EP.value
        if get_device() == "GPU" and CUDA_EP in self.had_providers:
            return True

        self.logger.warning(
            f"{CUDA_EP} is not in available providers ({self.had_providers}). Use {self.default_provider} inference by default."
        )

        install_instructions = [
            f"If you want to use {CUDA_EP} acceleration, you must do:"
            "(For reference only) If you want to use GPU acceleration, you must do:",
            "First, uninstall all onnxruntime packages in current environment.",
            "Second, install onnxruntime-gpu by `pip install onnxruntime-gpu`.",
            "Note the onnxruntime-gpu version must match your cuda and cudnn version.",
            "You can refer this link: https://onnxruntime.ai/docs/execution-providers/CUDA-EP.html",
            f"Third, ensure {CUDA_EP} is in available providers list. e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']",
        ]
        self.print_log(install_instructions)
        return False

    def is_dml_available(self) -> bool:
        if not self.cfg_use_dml:
            return False

        cur_os = platform.system()
        if cur_os != "Windows":
            self.logger.warning(
                f"DirectML is only supported in Windows OS. The current OS is {cur_os}. Use {self.default_provider} inference by default.",
            )
            return False

        window_build_number_str = platform.version().split(".")[-1]
        window_build_number = (
            int(window_build_number_str) if window_build_number_str.isdigit() else 0
        )
        if window_build_number < 18362:
            self.logger.warning(
                f"DirectML is only supported in Windows 10 Build 18362 and above OS. The current Windows Build is {window_build_number}. Use {self.default_provider} inference by default.",
            )
            return False

        DML_EP = EP.DIRECTML_EP.value
        if DML_EP in self.had_providers:
            return True

        self.logger.warning(
            f"{DML_EP} is not in available providers ({self.had_providers}). Use {self.default_provider} inference by default."
        )

        install_instructions = [
            f"{DML_EP} is not in available providers ({self.had_providers}). Use {self.default_provider} inference by default.",
            "If you want to use DirectML acceleration, you must do:",
            "First, uninstall all onnxruntime packages in current environment.",
            "Second, install onnxruntime-directml by `pip install onnxruntime-directml`",
            f"Third, ensure {DML_EP} is in available providers list. e.g. ['DmlExecutionProvider', 'CPUExecutionProvider']",
        ]
        self.print_log(install_instructions)
        return False

    def is_cann_available(self) -> bool:
        if not self.cfg_use_cann:
            return False

        CANN_EP = EP.CANN_EP.value
        if CANN_EP in self.had_providers:
            return True

        self.logger.warning(
            f"{CANN_EP} is not in available providers ({self.had_providers}). Use {self.default_provider} inference by default."
        )
        install_instructions = [
            "If you want to use CANN acceleration, you must do:",
            "First, ensure you have installed Huawei Ascend software stack.",
            "Second, install onnxruntime with CANN support by following the instructions at:",
            "\thttps://onnxruntime.ai/docs/execution-providers/CANN-ExecutionProvider.html",
            f"Third, ensure {CANN_EP} is in available providers list. e.g. ['CANNExecutionProvider', 'CPUExecutionProvider']",
        ]
        self.print_log(install_instructions)
        return False

    def print_log(self, log_list: List[str]):
        for log_info in log_list:
            self.logger.info(log_info)
