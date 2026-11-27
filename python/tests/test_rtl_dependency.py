# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import builtins
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.utils.typings import EngineType, LangRec


class _AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _to_attr_dict(obj):
    if isinstance(obj, dict):
        return _AttrDict({k: _to_attr_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_dict(v) for v in obj]
    return obj


def _to_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    return obj


def _module_unavailable(name: str) -> bool:
    return name not in sys.modules and importlib.util.find_spec(name) is None


def _install_missing_runtime_stubs(monkeypatch):
    if _module_unavailable("cv2"):
        monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))

    if _module_unavailable("colorlog"):
        colorlog_mod = types.ModuleType("colorlog")

        class ColoredFormatter:
            def __init__(self, *args, **kwargs):
                pass

        colorlog_mod.ColoredFormatter = ColoredFormatter
        monkeypatch.setitem(sys.modules, "colorlog", colorlog_mod)

    if _module_unavailable("omegaconf"):
        omegaconf_mod = types.ModuleType("omegaconf")

        class OmegaConf:
            @staticmethod
            def load(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return _to_attr_dict(yaml.safe_load(f))

            @staticmethod
            def to_container(cfg):
                return _to_plain_dict(cfg)

            @staticmethod
            def update(cfg, key, value):
                parts = key.split(".")
                cur = cfg
                for part in parts[:-1]:
                    cur = cur[part]
                cur[parts[-1]] = value

            @staticmethod
            def select(cfg, key):
                cur = cfg
                for part in key.split("."):
                    if not isinstance(cur, dict) or part not in cur:
                        return None
                    cur = cur[part]
                return cur

        omegaconf_mod.DictConfig = _AttrDict
        omegaconf_mod.OmegaConf = OmegaConf
        monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_mod)

    if _module_unavailable("pyclipper"):
        monkeypatch.setitem(sys.modules, "pyclipper", types.ModuleType("pyclipper"))

    if _module_unavailable("shapely"):
        shapely_mod = types.ModuleType("shapely")
        geom_mod = types.ModuleType("shapely.geometry")
        geom_mod.Polygon = object
        shapely_mod.geometry = geom_mod
        monkeypatch.setitem(sys.modules, "shapely", shapely_mod)
        monkeypatch.setitem(sys.modules, "shapely.geometry", geom_mod)


def _import_rtl_modules(monkeypatch):
    _install_missing_runtime_stubs(monkeypatch)
    from rapidocr import main as rapidocr_main
    from rapidocr.ch_ppocr_rec import main as rec_main
    from rapidocr.utils import utils
    from rapidocr.utils.model_resolver import normalize_lang

    return rapidocr_main, rec_main, utils, normalize_lang


def _block_bidi_import(monkeypatch):
    original_import = builtins.__import__

    def missing_bidi(name, *args, **kwargs):
        if name.startswith("bidi"):
            raise ModuleNotFoundError("No module named 'bidi'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_bidi)


def test_validate_rtl_dependency_reports_extra(monkeypatch):
    _, _, utils, _ = _import_rtl_modules(monkeypatch)
    _block_bidi_import(monkeypatch)

    with pytest.raises(ModuleNotFoundError, match=r"rapidocr\[rtl\]"):
        utils.validate_rtl_dependency()


def test_reorder_bidi_for_display_uses_bidi_algorithm(monkeypatch):
    _, _, utils, _ = _import_rtl_modules(monkeypatch)
    bidi_module = types.ModuleType("bidi")
    algorithm_module = types.ModuleType("bidi.algorithm")
    algorithm_module.get_display = lambda text: text[::-1]
    bidi_module.algorithm = algorithm_module
    monkeypatch.setitem(sys.modules, "bidi", bidi_module)
    monkeypatch.setitem(sys.modules, "bidi.algorithm", algorithm_module)

    assert utils.reorder_bidi_for_display(("abc", "123")) == ("cba", "321")


def test_arabic_dependency_is_checked_before_engine_initialization(monkeypatch):
    _, rec_main, _, _ = _import_rtl_modules(monkeypatch)
    engine_called = False

    def fail_get_engine(_engine_type):
        nonlocal engine_called
        engine_called = True
        raise AssertionError(
            "engine initialization must happen after dependency validation"
        )

    def missing_dependency():
        raise ModuleNotFoundError("RTL dependency is missing")

    monkeypatch.setattr(rec_main, "get_engine", fail_get_engine)
    monkeypatch.setattr(rec_main, "validate_rtl_dependency", missing_dependency)

    cfg = SimpleNamespace(lang_type=LangRec.ARABIC, engine_type=EngineType.ONNXRUNTIME)
    with pytest.raises(ModuleNotFoundError, match="RTL dependency is missing"):
        rec_main.TextRecognizer(cfg)

    assert engine_called is False


def test_arabic_enum_is_normalized_for_rtl_detection(monkeypatch):
    _, rec_main, _, normalize_lang = _import_rtl_modules(monkeypatch)
    assert normalize_lang(LangRec.ARABIC) in rec_main.RTL_LANGS


def test_rapidocr_init_validates_rtl_dependency_when_use_rec_enabled(monkeypatch):
    rapidocr_main, _, _, _ = _import_rtl_modules(monkeypatch)
    _block_bidi_import(monkeypatch)

    with pytest.raises(ModuleNotFoundError, match=r"rapidocr\[rtl\]"):
        rapidocr_main.RapidOCR(params={"Rec.lang_type": LangRec.ARABIC})


def test_rapidocr_validates_rtl_dependency_before_det_when_use_rec_enabled_on_call(
    monkeypatch,
):
    rapidocr_main, _, _, _ = _import_rtl_modules(monkeypatch)
    _block_bidi_import(monkeypatch)

    engine = rapidocr_main.RapidOCR(
        params={"Global.use_rec": False, "Rec.lang_type": LangRec.ARABIC}
    )

    def fail_det(*args, **kwargs):
        raise AssertionError("detection must not run when RTL dependency is missing")

    monkeypatch.setattr(engine, "detect_and_crop", fail_det)

    with pytest.raises(ModuleNotFoundError, match=r"rapidocr\[rtl\]"):
        engine(np.zeros((32, 32, 3), dtype=np.uint8), use_rec=True)
