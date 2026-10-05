# -*- encoding: utf-8 -*-
from typing import Any

from omegaconf import OmegaConf


def validate_model_routes(model_routes: Any) -> None:
    """Validate the engine-neutral route schema explicitly."""
    if not OmegaConf.is_dict(model_routes):
        raise ValueError("Invalid model_routes: expected a mapping")

    for path, routes in _iter_model_route_lists(model_routes):
        _validate_model_route_list(path, routes)


def _iter_model_route_lists(model_routes: Any):
    for version, version_cfg in model_routes.items():
        if version == "_defs":
            continue
        if not OmegaConf.is_dict(version_cfg):
            raise ValueError(f"Invalid model route version: {version}")
        if "aliases" in version_cfg:
            _validate_aliases(f"{version}.aliases", version_cfg["aliases"])

        for task, task_cfg in version_cfg.items():
            if task == "aliases":
                continue
            if not OmegaConf.is_dict(task_cfg):
                raise ValueError(f"Invalid model route task: {version}.{task}")
            if "aliases" in task_cfg:
                _validate_aliases(f"{version}.{task}.aliases", task_cfg["aliases"])

            for model_type, routes in task_cfg.items():
                if model_type != "aliases":
                    yield f"{version}.{task}.{model_type}", routes


def _validate_aliases(path: str, aliases: Any) -> None:
    if not OmegaConf.is_dict(aliases):
        raise ValueError(f"Invalid aliases: {path}")

    for alias, target in aliases.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError(f"Invalid alias: {path}")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"Invalid alias target: {path}.{alias}")


def _validate_model_route_list(path: str, routes: Any) -> None:
    if not OmegaConf.is_list(routes):
        raise ValueError(f"Invalid model route list: {path}")

    seen = set()
    seen_languages = set()
    for route in routes:
        if not OmegaConf.is_dict(route):
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
        if not OmegaConf.is_list(supported_langs) or not supported_langs:
            raise ValueError(f"Empty supported languages: {model_key}")

        for lang in supported_langs:
            if not isinstance(lang, str) or not lang.strip():
                raise ValueError(f"Invalid supported language: {model_key}")
            if lang in seen_languages:
                raise ValueError(f"Overlapping supported language: {path}.{lang}")
            seen_languages.add(lang)
