# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Union

import yaml

from ..utils.log import logger
from ..utils.model_resolver import normalize_lang, resolve_model_key
from ..utils.typings import ModelType

cur_dir = Path(__file__).resolve().parent.parent
MODEL_URL_PATH = cur_dir / "default_models.yaml"

# rapidocr 5 names the built-in interpreter PPOCR_CPP as well as ONNXRUNTIME.
# Both load the same ONNX files published under the onnxruntime catalog.
_ENGINE_ALIASES = {"ppocr_cpp": "onnxruntime"}


def _load_model_info(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping")
    return loaded


@dataclass
class FileInfo:
    engine_type: Enum
    ocr_version: Enum
    task_type: Enum
    lang_type: Union[Enum, str]
    model_type: Enum


class InferSession:
    model_info = _load_model_info(MODEL_URL_PATH)

    @classmethod
    def get_model_url(cls, file_info: FileInfo) -> Dict[str, str]:
        engine_type = file_info.engine_type.value
        engine_type = _ENGINE_ALIASES.get(engine_type, engine_type)
        ocr_version = file_info.ocr_version.value
        task_type = file_info.task_type.value
        lang_type = normalize_lang(file_info.lang_type)
        model_type = file_info.model_type.value

        model_dict = cls.model_info.get(engine_type, {}).get(ocr_version, {}).get(task_type)

        if not model_dict:
            raise ValueError(
                f"Unsupported configuration: {engine_type}.{ocr_version}.{task_type}.{model_type}"
            )

        model_key = resolve_model_key(
            file_info.task_type,
            file_info.ocr_version,
            file_info.lang_type,
            file_info.model_type,
        )

        if model_key is not None:
            if model_key in model_dict:
                return model_dict[model_key]

            raise ValueError(
                f"Unsupported configuration: {engine_type}.{ocr_version}.{task_type}.{lang_type}.{model_type}"
            )

        # 优先查找 server 模型
        if model_type == ModelType.SERVER.value:
            for key in model_dict:
                if key.startswith(lang_type) and model_type in key:
                    return model_dict[key]

        for key in model_dict:
            if key.startswith(lang_type) and model_type in key:
                return model_dict[key]

        logger.error(
            "Unsupported configuration:\n"
            f"  engine_type   = {engine_type}\n"
            f"  ocr_version   = {ocr_version}\n"
            f"  task_type     = {task_type}\n"
            f"  lang_type     = {lang_type}\n"
            f"  model_type     = {model_type}\n"
            "\n"
            "Please refer to the official model list for supported combinations:\n"
            "https://rapidai.github.io/RapidOCRDocs/main/model_list/\n"
            "\n"
            "Example valid usage:\n"
            "  from rapidocr import LangRec, OCRVersion, RapidOCR\n"
            "  engine = RapidOCR(params={'Rec.ocr_version': OCRVersion.PPOCRV5, 'Rec.lang_type': LangRec.CH, 'Rec.model_type': 'mobile'})",
        )
        raise ValueError("Invalid OCR configuration.")

    @classmethod
    def get_dict_key_url(cls, file_info: FileInfo) -> str:
        model_dict = cls.get_model_url(file_info)
        return model_dict["dict_url"]
