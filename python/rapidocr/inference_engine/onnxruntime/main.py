# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from ...utils import DownloadFile, DownloadFileInput
from ...utils.logger import Logger
from ..base import FileInfo, InferSession
from .provider_config import ProviderConfig


class OrtInferSession(InferSession):
    def __init__(self, cfg: Dict[str, Any]):
        self.logger = Logger(logger_name=__name__).get_log()

        # support custom session (PR #451)
        session = cfg.get("session", None)
        if session is not None:
            if not isinstance(session, InferenceSession):
                raise TypeError(
                    f"Expected session to be an InferenceSession, got {type(session)}"
                )

            self.logger.debug("Using the provided InferenceSession for inference.")
            self.session = session
            return

        model_path = cfg.get("model_path", None)
        if model_path is None:
            # 说明用户没有指定自己模型，使用默认模型
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
                logger=self.logger,
            )
            DownloadFile.run(download_params)

        self.logger.info(f"Using {model_path}")
        model_path = Path(model_path)
        self._verify_model(model_path)

        sess_opt = self._init_sess_opts(cfg.engine_cfg)

        provider_cfg = ProviderConfig(engine_cfg=cfg.engine_cfg)
        self.session = InferenceSession(
            model_path,
            sess_options=sess_opt,
            providers=provider_cfg.get_ep_list(),
        )
        provider_cfg.verify_providers(self.session.get_providers())

    @staticmethod
    def _init_sess_opts(cfg: Dict[str, Any]) -> SessionOptions:
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = cfg.enable_cpu_mem_arena
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cpu_nums = os.cpu_count()
        intra_op_num_threads = cfg.get("intra_op_num_threads", -1)
        if intra_op_num_threads != -1 and 1 <= intra_op_num_threads <= cpu_nums:
            sess_opt.intra_op_num_threads = intra_op_num_threads

        inter_op_num_threads = cfg.get("inter_op_num_threads", -1)
        if inter_op_num_threads != -1 and 1 <= inter_op_num_threads <= cpu_nums:
            sess_opt.inter_op_num_threads = inter_op_num_threads

        return sess_opt

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), [input_content]))
        try:
            return self.session.run(self.get_output_names(), input_dict)[0]
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(self) -> List[str]:
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character") -> List[str]:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in meta_dict.keys():
            return True
        return False


class ONNXRuntimeError(Exception):
    pass
