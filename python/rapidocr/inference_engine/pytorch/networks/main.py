# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ....utils.download_file import DownloadFile, DownloadFileInput
from ....utils.log import logger
from ...base import FileInfo, InferSession
from .architectures.base_model import BaseModel


class ModelLoader:
    root_dir = Path(__file__).resolve().parent
    DEFAULT_CFG_PATH = root_dir / "arch_config.yaml"

    def __init__(self, cfg, device: torch.device):
        model_path = self._init_model_path(cfg)
        arch_config = self._load_arch_config(model_path)

        self.predictor = self._build_and_load_model(arch_config, model_path)
        self.predictor.to(device)
        self.predictor.eval()

    def _init_model_path(self, cfg) -> Path:
        model_path = cfg.get("model_path", None)
        if model_path is None:
            model_info = InferSession.get_model_url(
                FileInfo(
                    engine_type=cfg.engine_type,
                    ocr_version=cfg.ocr_version,
                    task_type=cfg.task_type,
                    lang_type=cfg.lang_type,
                    model_type=cfg.model_type,
                )
            )
            default_model_url = model_info["model_dir"]
            model_path = InferSession.DEFAULT_MODEL_PATH / Path(default_model_url).name
            DownloadFile.run(
                DownloadFileInput(
                    file_url=default_model_url,
                    sha256=model_info["SHA256"],
                    save_path=model_path,
                    logger=logger,
                )
            )

        logger.info(f"Using {model_path}")
        InferSession._verify_model(model_path)
        return Path(model_path)

    def _load_arch_config(self, model_path: Path):
        all_arch_config = OmegaConf.load(self.DEFAULT_CFG_PATH)

        file_name = model_path.stem
        if file_name not in all_arch_config:
            raise ValueError(f"architecture {file_name} is not in arch_config.yaml")

        return all_arch_config.get(file_name)

    def _build_and_load_model(self, arch_config, model_path: Path):
        model = BaseModel(arch_config)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        return model
