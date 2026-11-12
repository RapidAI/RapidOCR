# -*- encoding: utf-8 -*-
"""Tests for document-family detection used by the OCR model router."""
import sys
from pathlib import Path

import cv2
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.utils.doc_family import (  # noqa: E402
    CAD,
    CLEAN,
    DEFAULT_MIN_LINE_LEN,
    count_long_lines,
    detect_doc_family,
)


def _blank(h=800, w=1000):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_clean_page_has_few_lines():
    img = _blank()
    cv2.putText(img, "CONSTANCIA DE TRABAJO", (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "El trabajador labora desde 2020", (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    family, n = detect_doc_family(img)
    assert family == CLEAN, n


def test_cad_page_has_many_lines():
    img = _blank()
    for y in range(10, 795, 15):
        cv2.line(img, (10, y), (990, y), (0, 0, 0), 1)
    for x in range(10, 995, 15):
        cv2.line(img, (x, 10), (x, 790), (0, 0, 0), 1)
    n = count_long_lines(img)
    family, _ = detect_doc_family(img)
    assert n >= 100, n
    assert family == CAD, n


def test_empty_page_is_clean():
    family, n = detect_doc_family(_blank())
    assert family == CLEAN
    assert n == 0


def test_default_line_len_is_long():
    # Real calibration: clean scans stay under 20 segments at 250 px.
    assert DEFAULT_MIN_LINE_LEN >= 200
