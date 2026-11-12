"""Per-document-type post-processing for Spanish OCR output.

Implements the deterministic correction layer described in SPANISH_MULTIDOC.md
section 4.7 and OPTIMIZATION_STRATEGY.md section 6:

- Regex constraint masking for structured fields (DNI, RUC, passport, amounts,
  dates, CAD dimensions).
- Homoglyph normalization in numeric slots (O->0, I->1, S->5, etc.).

These are pure functions operating on already-decoded text; they do not touch
the neural model.
"""

from __future__ import annotations

import re
from typing import Dict

_HOMOGLYPH_MAP = {
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "S": "5",
    "s": "5",
    "Z": "2",
    "z": "2",
    "B": "8",
    "G": "6",
    "T": "7",
}

PATTERNS: Dict[str, re.Pattern] = {
    "dni": re.compile(r"\b\d{8}\b"),
    "ruc": re.compile(r"\b\d{11}\b"),
    "passport": re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
    "amount": re.compile(r"\b\d{1,3}(?:\.\d{3})*,\d{2}\b"),
    "date_slash": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
    "date_dash": re.compile(r"\b\d{2}-\d{2}-\d{4}\b"),
    "cad_dim": re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|in|ft)\b", re.IGNORECASE),
}


def normalize_homoglyphs_in_numbers(text: str) -> str:
    def _fix_token(tok: str) -> str:
        if not any(ch.isdigit() for ch in tok):
            return tok
        return "".join(_HOMOGLYPH_MAP.get(ch, ch) for ch in tok)

    parts = re.split(r"(\s+)", text)
    return "".join(_fix_token(p) if not p.isspace() else p for p in parts)


def extract_fields(text: str) -> Dict[str, list]:
    result: Dict[str, list] = {}
    for name, pat in PATTERNS.items():
        result[name] = pat.findall(text)
    return result


def mask_text(text: str, doc_type: str) -> str:
    if doc_type in ("DNI", "RUC", "CARNET_RETCC", "LICENCIA_CONDUCIR", "FACTURA", "RECIBO", "VOUCHER"):
        text = normalize_homoglyphs_in_numbers(text)
    return text