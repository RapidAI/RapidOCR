"""Spanish OCR router: picks the recognition checkpoint per document family.

The CAD-tuned model wins on technical drawings; the baseline wins on clean
CV/CUL pages (see the per-domain benchmark in the strategy document). This
wrapper keeps both engines alive and routes each page by visual family.

Usage:
  from rapidocr.utils.model_router import SpanishOCRRouter

  router = SpanishOCRRouter()
  result = router("plano.png")          # -> CAD model
  result = router("cv.png")             # -> clean model
  family, lines = router.family_of("plano.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from rapidocr import EngineType, LangRec, RapidOCR

from .doc_family import CAD, detect_doc_family

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
# CAD model trained with the 107-char vocabulary; clean model with the extended
# 116-char vocabulary. Each engine must use the dict it was trained with, or the
# CTC indices decode as garbage.
DEFAULT_CAD_MODEL = MODELS_DIR / "spanish_rec_v18.onnx"
DEFAULT_CAD_DICT = MODELS_DIR / "ppocr_keys_es.txt"
DEFAULT_CLEAN_MODEL = MODELS_DIR / "spanish_rec_v18.onnx"
DEFAULT_CLEAN_DICT = MODELS_DIR / "ppocr_keys_es.txt"


def _make_engine(model_path: Path, dict_path: Path, params: Optional[Dict[str, Any]] = None) -> RapidOCR:
    merged: Dict[str, Any] = {
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.model_path": model_path,
        "Rec.rec_keys_path": dict_path,
        "Rec.lang_type": LangRec.ES,
    }
    if params:
        merged.update(params)
    return RapidOCR(params=merged)


class SpanishOCRRouter:
    def __init__(
        self,
        cad_model: Path = DEFAULT_CAD_MODEL,
        cad_dict: Path = DEFAULT_CAD_DICT,
        clean_model: Path = DEFAULT_CLEAN_MODEL,
        clean_dict: Path = DEFAULT_CLEAN_DICT,
        params: Optional[Dict[str, Any]] = None,
        min_lines: int = 100,
        min_line_len: int = 250,
    ):
        self.cad_model = Path(cad_model)
        self.cad_dict = Path(cad_dict)
        self.clean_model = Path(clean_model)
        self.clean_dict = Path(clean_dict)
        self.params = params
        self.min_lines = min_lines
        self.min_line_len = min_line_len
        self._cad = None
        self._clean = None

    def _engine(self, family: str) -> RapidOCR:
        if family == CAD:
            if self._cad is None:
                self._cad = _make_engine(self.cad_model, self.cad_dict, self.params)
            return self._cad
        if self._clean is None:
            self._clean = _make_engine(self.clean_model, self.clean_dict, self.params)
        return self._clean

    def family_of(self, img_or_path) -> tuple[str, int]:
        img = cv2.imread(str(img_or_path)) if isinstance(img_or_path, (str, Path)) else img_or_path
        return detect_doc_family(img, self.min_lines, self.min_line_len)

    def __call__(self, img_or_path, **kwargs):
        family, _ = self.family_of(img_or_path)
        return self._engine(family)(img_or_path, **kwargs)