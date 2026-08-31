# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.ch_ppocr_rec import main as rec_main
from rapidocr.utils import utils
from rapidocr.utils.model_resolver import normalize_lang
from rapidocr.utils.typings import EngineType, LangRec


def test_validate_rtl_dependency_reports_extra(monkeypatch):
    original_import = builtins.__import__

    def missing_bidi(name, *args, **kwargs):
        if name.startswith("bidi"):
            raise ModuleNotFoundError("No module named 'bidi'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_bidi)

    with pytest.raises(ModuleNotFoundError, match=r"rapidocr\[rtl\]"):
        utils.validate_rtl_dependency()


def test_reorder_bidi_for_display_uses_bidi_algorithm(monkeypatch):
    bidi_module = types.ModuleType("bidi")
    algorithm_module = types.ModuleType("bidi.algorithm")
    algorithm_module.get_display = lambda text: text[::-1]
    bidi_module.algorithm = algorithm_module
    monkeypatch.setitem(sys.modules, "bidi", bidi_module)
    monkeypatch.setitem(sys.modules, "bidi.algorithm", algorithm_module)

    assert utils.reorder_bidi_for_display(("abc", "123")) == ("cba", "321")


def test_arabic_dependency_is_checked_before_engine_initialization(monkeypatch):
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


def test_arabic_enum_is_normalized_for_rtl_detection():
    assert normalize_lang(LangRec.ARABIC) in rec_main.RTL_LANGS
