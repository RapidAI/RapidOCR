# -*- encoding: utf-8 -*-
"""Tests for Spanish per-type post-processing."""
import sys
from pathlib import Path


root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.utils.postprocess_es import (  # noqa: E402
    extract_fields,
    mask_text,
    normalize_homoglyphs_in_numbers,
)


def test_normalize_homoglyphs():
    assert normalize_homoglyphs_in_numbers("DNI O1234567") == "DNI 01234567"
    # prose untouched
    assert normalize_homoglyphs_in_numbers("hola mundo") == "hola mundo"


def test_extract_dni():
    fields = extract_fields("El DNI es 12345678 y fecha 01/02/2020")
    assert "12345678" in fields["dni"]
    assert "01/02/2020" in fields["date_slash"]


def test_extract_amount():
    fields = extract_fields("Total: 1.234,56 soles")
    assert "1.234,56" in fields["amount"]


def test_extract_cad_dim():
    fields = extract_fields("Ancho 12.5 mm alto 3 cm")
    assert len(fields["cad_dim"]) >= 2


def test_mask_text_dni():
    out = mask_text("DNI O1234567", "DNI")
    assert "01234567" in out


def test_mask_text_non_id_unchanged():
    out = mask_text("hola O mundo", "INFORME")
    assert out == "hola O mundo"


def test_ruc_pattern():
    fields = extract_fields("RUC 20123456789")
    assert "20123456789" in fields["ruc"]
