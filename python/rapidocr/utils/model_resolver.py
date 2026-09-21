# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import OmegaConf

from .typings import ModelType, OCRVersion, TaskType

MODEL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "default_models.yaml"
_MODEL_CONFIG = OmegaConf.load(MODEL_CONFIG_PATH)
MODEL_ROUTES = _MODEL_CONFIG.get("model_routes", {})


def route_to_model_key(
    task_type: TaskType,
    ocr_version: OCRVersion,
    lang_type: Union[Enum, str],
    model_type: ModelType,
) -> Optional[str]:
    task_cfg = _get_task_config(task_type, ocr_version)
    if task_cfg is None:
        return None

    lang_type = normalize_lang(lang_type)
    lang = _get_aliases(task_cfg, ocr_version).get(lang_type, lang_type)

    routes = task_cfg.get(model_type.value)
    if routes is None:
        return None

    for route in routes:
        supported_langs = route.get("supported_langs", [])
        if lang in supported_langs:
            return route["model_key"]

    raise ValueError(
        f"Unsupported {task_type.value}.lang_type={lang!r} for {ocr_version.value} {model_type.value} model. "
        f"Supported languages: {', '.join(_supported(routes))}"
    )


def list_supported_langs(
    task_type: TaskType, ocr_version: OCRVersion, model_type: Optional[ModelType] = None
) -> List[str]:
    task_cfg = _get_task_config(task_type, ocr_version)
    if task_cfg is None:
        return []

    if model_type is not None:
        return _supported(task_cfg.get(model_type.value))

    values = set()
    for key, routes in task_cfg.items():
        if key != "aliases":
            values.update(_supported(routes))

    return sorted(values)


def normalize_lang(lang_type: Union[Enum, str]) -> str:
    value = lang_type.value if isinstance(lang_type, Enum) else lang_type
    return str(value).strip().lower()


def _get_task_config(task_type: TaskType, ocr_version: OCRVersion) -> Any:
    version_cfg = MODEL_ROUTES.get(ocr_version.value)
    return version_cfg.get(task_type.value) if version_cfg else None


def _get_aliases(task_cfg: Any, ocr_version: OCRVersion) -> Dict[str, str]:
    version_cfg = MODEL_ROUTES.get(ocr_version.value)
    aliases = dict((version_cfg or {}).get("aliases", {}) or {})
    aliases.update(dict((task_cfg or {}).get("aliases", {}) or {}))
    return aliases


def _supported(routes: Any) -> List[str]:
    values = set()
    for route in routes or []:
        supported_langs = route.get("supported_langs", [])
        values.update(str(x) for x in supported_langs)
    return sorted(values)

