"""Table and formula extraction using PaddleOCR PPStructureV3.

Provides:
- extract_tables(img): list of HTML strings for detected tables.
- extract_formulas(img): list of LaTeX strings for detected formulas.
- extract_all(img): both, running the pipeline only once.

PPStructureV3 bundles layout detection + table structure + formula recognition.
Models download on first use. NOTE: requires PaddlePaddle >= 3.0 built with the
onednn fix for `pir::ArrayAttribute<pir::DoubleAttribute>` (see tools/build_paddle.sh).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

_ppstructure = None


def _get_structure():
    global _ppstructure
    if _ppstructure is None:
        from paddleocr import PPStructureV3

        kwargs: Dict[str, Any] = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_table_recognition=True,
            use_formula_recognition=True,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_region_detection=False,
        )
        # Point the table/text recognizer at the Spanish model when configured.
        # The directory must come from tools/make_ppstructure_rec.py (model_name
        # rewritten to the name PaddleX expects).
        rec_dir = os.getenv("PPSTRUCTURE_REC_DIR")
        if rec_dir:
            kwargs["text_recognition_model_dir"] = rec_dir
        _ppstructure = PPStructureV3(**kwargs)
    return _ppstructure


def _as_dict(result: Any) -> Dict[str, Any]:
    """Normalize a PaddleX Result object to its plain dict payload."""
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        try:
            payload = dict(result)
        except Exception:
            return {}
    if isinstance(payload.get("res"), dict):
        payload = payload["res"]
    return payload


def clean_formula(latex: str) -> str:
    """Strip recognition artifacts the formula model appends on tight crops.

    Observed on real pages: a trailing CJK glyph (`一`) or a dangling `=`
    that is not part of the expression. Only trailing/leading noise is
    removed; the expression body is never touched.
    """
    text = (latex or "").strip()
    while True:
        stripped = text.rstrip().rstrip("=").rstrip()
        stripped = stripped.rstrip("。、，；：")
        if stripped and "\u4e00" <= stripped[-1] <= "\u9fff":
            stripped = stripped[:-1].rstrip()
        if stripped == text:
            break
        text = stripped
    return text


def _parse(img_path: str) -> Tuple[List[str], List[str]]:
    tables: List[str] = []
    formulas: List[str] = []
    for result in _get_structure().predict(img_path):
        payload = _as_dict(result)
        for block in payload.get("parsing_res_list", []) or []:
            label = str(block.get("block_label", "")).lower()
            content = block.get("block_content") or ""
            if not content:
                continue
            if "table" in label:
                tables.append(str(content))
            elif "formula" in label:
                cleaned = clean_formula(str(content))
                if cleaned:
                    formulas.append(cleaned)
    return tables, formulas


def extract_tables(img_path: str) -> List[str]:
    """Return list of HTML strings for tables detected in the image."""
    return _parse(img_path)[0]


def extract_formulas(img_path: str) -> List[str]:
    """Return list of LaTeX strings for formulas detected in the image."""
    return _parse(img_path)[1]


def extract_all(img_path: str) -> Tuple[List[str], List[str]]:
    """Return (tables_html, formulas_latex), running the pipeline once."""
    return _parse(img_path)