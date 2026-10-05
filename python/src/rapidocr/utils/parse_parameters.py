"""Dotted-parameter config, with the same load/update entry points as 3.x.

The file format is the ``config.yaml`` shipped on ``main``. Values are a
nested dict that also supports attribute access (``cfg.Det.thresh``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..typings import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion, TaskType

_PACKAGE_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"

_ENUM_BY_KEY = {
    "engine_type": EngineType,
    "ocr_version": OCRVersion,
    "model_type": ModelType,
    "task_type": TaskType,
}
_LANG_BY_SECTION = {"Det": LangDet, "Rec": LangRec, "Cls": LangCls}


class ConfigNode(dict):
    """Dict that also supports ``node.key`` reads and writes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
        super().__setitem__(key, value)

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        super().__init__()
        if not data:
            return
        for key, value in data.items():
            self[key] = value


def _coerce(section: Optional[str], key: str, value: Any) -> Any:
    if isinstance(value, str):
        enum = _ENUM_BY_KEY.get(key)
        if key == "lang_type":
            enum = _LANG_BY_SECTION.get(section or "", enum)
        if enum is not None:
            try:
                return enum(value)
            except ValueError:
                return value
    return value


def _coerce_tree(node: ConfigNode, section: Optional[str] = None) -> None:
    for key, value in list(node.items()):
        if isinstance(value, ConfigNode):
            child_section = key if key in _LANG_BY_SECTION else section
            _coerce_tree(value, child_section)
            continue
        node[key] = _coerce(section, key, value)


class ParseParams:
    @staticmethod
    def load(config_path: Optional[str] = None) -> ConfigNode:
        path = Path(config_path) if config_path else _PACKAGE_CONFIG
        if not path.is_file():
            raise FileNotFoundError(str(path))
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("config root must be a mapping")
        node = ConfigNode(loaded)
        _coerce_tree(node)
        return node

    @staticmethod
    def update_batch(cfg: ConfigNode, params: Mapping[str, Any]) -> ConfigNode:
        for key, value in params.items():
            parts = str(key).split(".")
            if len(parts) < 2 or not all(parts):
                raise ValueError(f"{key} is not a valid key.")
            if parts[0] not in cfg:
                raise ValueError(f"{key} is not a valid key.")
            node = cfg[parts[0]]
            if not isinstance(node, ConfigNode):
                raise ValueError(f"{key} is not a valid key.")
            section = parts[0]
            for part in parts[1:-1]:
                child = node.get(part)
                if not isinstance(child, ConfigNode):
                    raise ValueError(f"{key} is not a valid key.")
                node = child
            leaf = parts[-1]
            if hasattr(value, "value") and not isinstance(value, (str, bytes)):
                stored = value
            else:
                stored = _coerce(section, leaf, value)
            node[leaf] = stored
        return cfg
