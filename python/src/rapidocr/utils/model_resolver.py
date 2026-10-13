# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
"""Route (task, version, lang, model_type) to a model key.

The routes live in ``default_models.yaml`` under ``model_routes`` so the
rapidocr 5 package and the 3.x package on ``main`` share one schema.
"""
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..typings import ModelType, OCRVersion, TaskType

MODEL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "default_models.yaml"


def _load_model_config(path: Path) -> Dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping")
    return loaded


_MODEL_CONFIG = _load_model_config(MODEL_CONFIG_PATH)
MODEL_ROUTES: Dict[str, Any] = _MODEL_CONFIG.get("model_routes", {}) or {}


def load_lang_mapping(lang_mapping: Dict[str, Any]) -> Dict[str, Any]:
    flat_map: Dict[str, Any] = {}
    for main_code, info in lang_mapping.items():
        cn = info["cn"]
        en = info["en"]
        aliases = info.get("aliases", []) or []

        if main_code in flat_map:
            raise ValueError(f"Duplicate definition for main code {main_code}!")
        flat_map[main_code] = {"cn": cn, "en": en}

        for alias in aliases:
            if alias in flat_map:
                raise ValueError(f"Conflict: alias {alias} already exists!")
            flat_map[alias] = {"cn": cn, "en": en}
    return flat_map


LANG_MAPPING = load_lang_mapping(_MODEL_CONFIG.get("lang_mapping", {}) or {})


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
        if lang in [str(x) for x in route.get("supported_langs", []) or []]:
            return route["model_key"]

    supported_langs = [
        f"{x} ({LANG_MAPPING.get(x, {}).get('en', x)})" for x in _supported(routes)
    ]
    raise ValueError(
        f"Unsupported {task_type.value}.lang_type={lang!r} for {ocr_version.value} "
        f"{model_type.value} model. Supported languages: {', '.join(supported_langs)}"
    )


# Name used by earlier rapidocr 5 alphas.
resolve_model_key = route_to_model_key


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
        values.update(str(x) for x in route.get("supported_langs", []) or [])
    return sorted(values)
