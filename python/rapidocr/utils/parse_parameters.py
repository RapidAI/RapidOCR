# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union

from omegaconf import DictConfig, OmegaConf

from .typings import (
    EngineType,
    LangCls,
    LangDet,
    LangRec,
    ModelType,
    OCRVersion,
    TaskType,
)


class ParseParams(OmegaConf):
    def __init__(self):
        pass

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> DictConfig:
        cfg = OmegaConf.load(file_path)

        cfg.Det = cls._convert_value_to_enum(cfg.Det)
        cfg.Cls = cls._convert_value_to_enum(cfg.Cls)
        cfg.Rec = cls._convert_value_to_enum(cfg.Rec)
        return cfg

    @classmethod
    def update_batch(cls, cfg: DictConfig, params: Dict[str, Any]) -> DictConfig:
        global_keys = list(OmegaConf.to_container(cfg.Global).keys())
        section_keys = set(OmegaConf.to_container(cfg).keys())
        enum_params = [
            "engine_type",
            "model_type",
            "ocr_version",
            "lang_type",
            "task_type",
        ]
        for k, v in params.items():
            key_parts = k.split(".")
            if len(key_parts) < 2 or not all(key_parts):
                raise ValueError(f"{k} is not a valid key.")

            section = key_parts[0]
            if section not in section_keys:
                raise ValueError(f"{k} is not a valid key.")

            if section == "Global" and (
                len(key_parts) != 2 or key_parts[1] not in global_keys
            ):
                raise ValueError(f"{k} is not a valid key.")

            param_name = key_parts[-1]
            if param_name in enum_params and not isinstance(v, Enum):
                raise TypeError(f"The value of {k} must be Enum Type.")

            cls.update(cfg, k, v)
        return cfg

    @classmethod
    def _convert_value_to_enum(cls, cfg: DictConfig):
        cfg.engine_type = EngineType(cfg.engine_type)
        cfg.model_type = ModelType(cfg.model_type)
        cfg.ocr_version = OCRVersion(cfg.ocr_version)
        cfg.task_type = TaskType(cfg.task_type)
        cfg.lang_type = cls.LangType(cfg.task_type, cfg.lang_type)
        return cfg

    @staticmethod
    def LangType(task_type: TaskType, lang_type: str):
        if task_type == TaskType.DET:
            return LangDet(lang_type)

        if task_type == TaskType.CLS:
            return LangCls(lang_type)

        if task_type == TaskType.REC:
            return LangRec(lang_type)

        raise ValueError(f"task_type {task_type.value} is not in [Det, Cls, Rec]")
