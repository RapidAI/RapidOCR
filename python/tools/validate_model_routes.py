# -*- encoding: utf-8 -*-
from typing import Any

from omegaconf import OmegaConf


def validate_model_routes(model_routes: Any) -> None:
    """Validate the engine-neutral route schema explicitly."""
    for path, routes in _iter_model_route_lists(model_routes):
        _validate_model_route_list(path, routes)


def _iter_model_route_lists(model_routes: Any):
    for version, version_cfg in model_routes.items():
        if version == "_defs":
            continue

        for task, task_cfg in version_cfg.items():
            if task == "aliases":
                continue

            for model_type, routes in task_cfg.items():
                if model_type != "aliases":
                    yield f"{version}.{task}.{model_type}", routes


def _validate_model_route_list(path: str, routes: Any) -> None:
    if not OmegaConf.is_list(routes):
        raise ValueError(f"Invalid model route list: {path}")

    seen = set()
    for route in routes:
        required = {"lang", "model_key", "supported_langs"}
        if not required.issubset(route):
            raise ValueError(f"Invalid model route entry: {path}")

        model_key = route["model_key"]
        if model_key in seen:
            raise ValueError(f"Duplicate model route key: {model_key}")

        seen.add(model_key)

        if not route["supported_langs"]:
            raise ValueError(f"Empty supported languages: {model_key}")
