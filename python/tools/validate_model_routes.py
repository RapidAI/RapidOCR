# -*- encoding: utf-8 -*-
from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import OmegaConf
from rapidocr.utils.typings import ModelType, OCRVersion, TaskType


def validate_model_routes(model_routes: Any) -> None:
    """Validate the engine-neutral route schema explicitly."""
    if not _is_mapping(model_routes):
        raise ValueError("Invalid model_routes: expected a mapping")

    for path, routes in _iter_model_route_lists(model_routes):
        _validate_model_route_list(path, routes)


def _iter_model_route_lists(model_routes: Any):
    for version, version_cfg in model_routes.items():
        if version == "_defs":
            continue
        _validate_enum("OCR version", version, OCRVersion)
        if not _is_mapping(version_cfg):
            raise ValueError(f"Invalid model route version: {version}")
        if "aliases" in version_cfg:
            _validate_aliases(f"{version}.aliases", version_cfg["aliases"])

        for task, task_cfg in version_cfg.items():
            if task == "aliases":
                continue
            _validate_enum("task", task, TaskType)
            if not _is_mapping(task_cfg):
                raise ValueError(f"Invalid model route task: {version}.{task}")
            if "aliases" in task_cfg:
                _validate_aliases(f"{version}.{task}.aliases", task_cfg["aliases"])

            for model_type, routes in task_cfg.items():
                if model_type != "aliases":
                    _validate_enum("model type", model_type, ModelType)
                    yield f"{version}.{task}.{model_type}", routes


def _validate_enum(label: str, value: str, enum_type: Any) -> None:
    try:
        enum_type(value)
    except ValueError as exc:
        valid_values = ", ".join(item.value for item in enum_type)
        raise ValueError(
            f"Invalid {label} {value!r}; expected one of: {valid_values}"
        ) from exc


def _is_mapping(value: Any) -> bool:
    return OmegaConf.is_dict(value) or isinstance(value, Mapping)


def _is_sequence(value: Any) -> bool:
    return OmegaConf.is_list(value) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    )


def _validate_aliases(path: str, aliases: Any) -> None:
    if not _is_mapping(aliases):
        raise ValueError(f"Invalid aliases: {path}")

    for alias, target in aliases.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError(f"Invalid alias: {path}")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"Invalid alias target: {path}.{alias}")


def _validate_model_route_list(path: str, routes: Any) -> None:
    if not _is_sequence(routes):
        raise ValueError(f"Invalid model route list: {path}")

    seen = set()
    seen_languages = set()
    for route in routes:
        if not _is_mapping(route):
            raise ValueError(f"Invalid model route entry: {path}")

        required = {"lang", "model_key", "supported_langs"}
        if not required.issubset(route):
            raise ValueError(f"Invalid model route entry: {path}")

        if not isinstance(route["lang"], str) or not route["lang"].strip():
            raise ValueError(f"Invalid route language: {path}")

        model_key = route["model_key"]
        if not isinstance(model_key, str) or not model_key.strip():
            raise ValueError(f"Invalid route model key: {path}")
        if model_key in seen:
            raise ValueError(f"Duplicate model route key: {model_key}")

        seen.add(model_key)

        supported_langs = route["supported_langs"]
        if not _is_sequence(supported_langs) or not supported_langs:
            raise ValueError(f"Empty supported languages: {model_key}")

        for lang in supported_langs:
            if not isinstance(lang, str) or not lang.strip():
                raise ValueError(f"Invalid supported language: {model_key}")
            if lang in seen_languages:
                raise ValueError(f"Overlapping supported language: {path}.{lang}")
            seen_languages.add(lang)
