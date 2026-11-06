# -*- encoding: utf-8 -*-
"""Tests for the Spanish document-type classifier."""
import sys
from pathlib import Path

import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.utils.doc_classifier import DEFAULT_TAXONOMY, classify  # noqa: E402


def test_classify_by_filename_hint():
    r = classify("", "cv_juan.pdf")
    assert r["type"] == "CV"
    assert r["confidence"] >= 50


def test_classify_by_keywords():
    text = "experiencia laboral educación habilidades referencias perfil profesional"
    r = classify(text, "archivo.pdf")
    assert r["type"] == "CV"
    assert r["scores"].get("CV", 0) > 0


def test_classify_dni():
    text = "documento nacional dni reniec fecha de nacimiento lugar de nacimiento"
    r = classify(text, "dni.pdf")
    assert r["type"] == "DNI"


def test_classify_factura():
    text = "factura ruc igv subtotal total a pagar sunat"
    r = classify(text, "factura_001.pdf")
    assert r["type"] == "FACTURA"


def test_classify_blueprint_visual_hint():
    r = classify("plano escala cota", "plano.pdf", visual_hints={"has_cad_lines": True})
    assert r["type"] == "BLUEPRINT"


def test_classify_fallback():
    r = classify("texto sin ninguna pista", "archivo.pdf")
    assert r["type"] == "OTRO"
    assert r["confidence"] == 0


def test_classify_missing_taxonomy_raises():
    with pytest.raises(ValueError):
        classify("x", "y", taxonomy={"document_types": []})


def test_default_taxonomy_has_all_families():
    types = set(DEFAULT_TAXONOMY["document_types"])
    for t in ("CV", "DNI", "BLUEPRINT", "FACTURA", "OTRO"):
        assert t in types
