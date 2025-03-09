# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import DictConfig
from openvino.runtime import Core

from ..utils import Logger, download_file
from .base import InferSession


class OpenVINOInferSession(InferSession):
    def __init__(self, config: DictConfig, mode: Optional[str] = None):
        super().__init__(config)
        self.mode = mode
        self.logger = Logger(logger_name=__name__).get_log()

        core = Core()

        model_path = config.get("model_path", None)
        if model_path is None:
            default_model_url = self.get_model_url(
                config.engine_name, config.task_type, config.lang
            )
            if self.mode == "rec":
                default_model_url = default_model_url.model_dir

            model_path = self.DEFAULT_MODE_PATH / Path(default_model_url).name
            download_file(default_model_url, model_path, self.logger)

        self._verify_model(model_path)
        model_onnx = core.read_model(model_path)

        cpu_nums = os.cpu_count()
        infer_num_threads = config.get("inference_num_threads", -1)
        if infer_num_threads != -1 and 1 <= infer_num_threads <= cpu_nums:
            core.set_property("CPU", {"INFERENCE_NUM_THREADS": str(infer_num_threads)})

        compile_model = core.compile_model(model=model_onnx, device_name="CPU")
        self.session = compile_model.create_infer_request()

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        try:
            self.session.infer(inputs=[input_content])
            return self.session.get_output_tensor().data
        except Exception as e:
            error_info = traceback.format_exc()
            raise OpenVIONError(error_info) from e

    def have_key(self, key: str = "character") -> bool:
        return False


class OpenVIONError(Exception):
    pass
