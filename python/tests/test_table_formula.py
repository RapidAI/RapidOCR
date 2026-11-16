# -*- encoding: utf-8 -*-
"""Tests for table/formula extraction wrapper."""
import sys
from pathlib import Path


root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.utils import table_formula  # noqa: E402
from rapidocr.utils.table_formula import (  # noqa: E402
    _as_dict,
    clean_formula,
    extract_all,
    extract_formulas,
    extract_tables,
)


class _FakeResult:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakePipeline:
    def __init__(self, payload):
        self._payload = payload

    def predict(self, _img_path):
        return [_FakeResult(self._payload)]


def _payload(*blocks):
    return {
        "res": {
            "parsing_res_list": [
                {"block_label": label, "block_content": content}
                for label, content in blocks
            ]
        }
    }


def test_imports():
    assert callable(extract_tables)
    assert callable(extract_formulas)
    assert callable(extract_all)


def test_as_dict_handles_callable_json():
    payload = _payload(("text", "hola"))
    assert _as_dict(_FakeResult(payload)) == payload["res"]


def test_parse_extracts_tables_and_formulas(monkeypatch):
    payload = _payload(
        ("table", "<html><body><table></table></body></html>"),
        ("formula", r"R_{n}=F_{y}A_{g}"),
        ("text", "texto normal"),
        ("footer", "pie"),
    )
    monkeypatch.setattr(table_formula, "_ppstructure", _FakePipeline(payload))
    tables, formulas = extract_all("dummy.png")
    assert tables == ["<html><body><table></table></body></html>"]
    assert formulas == [r"R_{n}=F_{y}A_{g}"]


def test_parse_ignores_non_table_formula_blocks(monkeypatch):
    payload = _payload(("text", "solo texto"), ("header", "cabecera"))
    monkeypatch.setattr(table_formula, "_ppstructure", _FakePipeline(payload))
    assert extract_tables("dummy.png") == []
    assert extract_formulas("dummy.png") == []


def test_parse_empty_payload(monkeypatch):
    monkeypatch.setattr(table_formula, "_ppstructure", _FakePipeline({}))
    assert extract_all("dummy.png") == ([], [])

def test_clean_formula_strips_trailing_cjk_glyph():
    assert clean_formula(r"R_{n}=F_{y}A_{g}一") == r"R_{n}=F_{y}A_{g}"


def test_clean_formula_strips_dangling_equals():
    assert clean_formula(r"\phi M_{n}=\phi F_{y}Z_{x}\geq M_{u}=") == r"\phi M_{n}=\phi F_{y}Z_{x}\geq M_{u}"


def test_clean_formula_keeps_valid_expression():
    assert clean_formula(r"x = \frac{a}{b}") == r"x = \frac{a}{b}"


def test_clean_formula_empty():
    assert clean_formula("") == ""
