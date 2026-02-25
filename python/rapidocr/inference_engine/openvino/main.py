# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import traceback
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

try:
    from openvino import Core
except ImportError:
    from openvino.runtime import Core

from ...utils.download_file import DownloadFile, DownloadFileInput
from ...utils.log import logger
from ..base import FileInfo, InferSession
from .device_config import CPUConfig


class OpenVINOInferSession(InferSession):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        core = Core()

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
            model_path = self.DEFAULT_MODEL_PATH / Path(model_info["model_dir"]).name
            download_params = DownloadFileInput(
                file_url=model_info["model_dir"],
                sha256=model_info["SHA256"],
                save_path=model_path,
                logger=logger,
            )
            DownloadFile.run(download_params)

        logger.info(f"Using {model_path}")
        model_path = Path(model_path)
        self._verify_model(model_path)

        # Use dedicated config class
        cpu_config = CPUConfig(cfg.get("engine_cfg", {}))
        core.set_property("CPU", cpu_config.get_config())

        self.model = core.read_model(model_path)
        compile_model = core.compile_model(model=self.model, device_name="CPU")
        self.session = compile_model.create_infer_request()

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        try:
            self.session.infer(inputs=[input_content])
            return self.session.get_output_tensor().data
        except Exception as e:
            error_info = traceback.format_exc()
            raise OpenVINOError(error_info) from e

    def have_key(self, key: str = "character") -> bool:
        try:
            self.get_character_list(key)
            return True
        except OpenVINOError:
            return False

    def get_character_list(self, key: str = "character") -> List[str]:
        framework_info = self.get_rt_info_framework()
        if framework_info is None:
            raise OpenVINOError(f"Failed to get runtime framework info")

        if key not in framework_info:
            raise OpenVINOError(f"Key '{key}' not found in framework info")

        val = framework_info[key]
        if not hasattr(val, "value"):
            raise OpenVINOError(
                f"Invalid value object for key '{key}': missing 'value' attribute"
            )

        value = getattr(val, "value", None)
        if value is None:
            raise OpenVINOError(f"Value is None for key '{key}'")

        return value.splitlines()

    def get_rt_info_framework(self):
        rt_info = self.model.get_rt_info()
        if "framework" not in rt_info:
            return None
        return rt_info["framework"]


class OpenVINOError(Exception):
    pass
