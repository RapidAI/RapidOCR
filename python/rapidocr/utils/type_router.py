"""Per-document-type recognizer routing.

Two checkpoints trained on the same pipeline beat each other on different paper
types (see the per-type table in the strategy document): one is better on CVs and
certificates, the other on receipts, family records and ESSALUD slips. Routing by
document type takes the best of both instead of averaging them, and differs from
per-crop routing in one important way: the type of a document is usually known or
classifiable, while picking a model per crop needs a rule that does not hold in
practice.

The type comes from, in order of preference:

1. what the caller states (`doc_type=`), typically from the source system;
2. the file name (`utils.doc_classifier` knows the usual conventions);
3. the recognized text, once a first pass has read the page.

Unknown types fall back to the default checkpoint, so the router is never worse
than the model it replaces.

Usage:
  from rapidocr.utils.type_router import TypeModelRouter

  router = TypeModelRouter()
  result = router("dni.pdf", doc_type="DNI")      # receipt-tuned checkpoint
  result = router("cv.pdf")                        # -> default checkpoint
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rapidocr import EngineType, LangRec, RapidOCR

from .doc_classifier import classify as classify_document

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = MODELS_DIR / "spanish_rec_v18.onnx"
ALT_MODEL = MODELS_DIR / "spanish_rec_v19.onnx"
DEFAULT_DICT = MODELS_DIR / "ppocr_keys_es.txt"

# Document types where the alternative checkpoint measured a lower CER, with the
# character-error rate of each model on the held-out gold that backs the choice.
# Types where the alternative checkpoint has a lower CER on the held-out gold.
# The figures are the standard character-weighted CER (edit distance over
# reference characters) that `tools/compare_routed.py` reports, because the mean
# of per-crop error rates (macro) tells a different story: on DNI and ESSALUD the
# alternative wins macro and loses micro, so a table built on macro numbers
# routes those types to the worse model.
ALT_MODEL_TYPES: Dict[str, str] = {
    "EXPERIENCIA": "0.0823 vs 0.0846",
    "RECORD_CONDUCTOR": "0.0823 vs 0.0846",
    "REGISTRO_EXPERIENCIA": "0.0823 vs 0.0846",
    "FAMILIAR": "0.1290 vs 0.1367",
    "DOCUMENTOS_FAMILIARES": "0.1290 vs 0.1367",
    "LICENCIA": "0.0247 vs 0.0256",
    "LICENCIA_CONDUCIR": "0.0247 vs 0.0256",
    "MIXTO": "0.0442 vs 0.0463",
    "OTROS": "0.0421 vs 0.0443",
    "VIDA_LEY": "0.1095 vs 0.1148",
    "VOUCHER": "0.0248 vs 0.0312",
    # Types deliberately left on the default checkpoint: CV, CERTIFICADO,
    # PLANOS, ACTA, DNI, CARNET, ESSALUD.
}


def _build(model_path: Path, dict_path: Path, params: Optional[Dict[str, Any]]) -> RapidOCR:
    merged: Dict[str, Any] = {
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.model_path": model_path,
        "Rec.rec_keys_path": dict_path,
        "Rec.lang_type": LangRec.ES,
    }
    if params:
        merged.update(params)
    return RapidOCR(params=merged)


class TypeModelRouter:
    def __init__(
        self,
        default_model: Path = DEFAULT_MODEL,
        alt_model: Path = ALT_MODEL,
        dict_path: Path = DEFAULT_DICT,
        alt_types: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.default_model = Path(default_model)
        self.alt_model = Path(alt_model)
        self.dict_path = Path(dict_path)
        self.alt_types = dict(alt_types or ALT_MODEL_TYPES)
        self.params = params
        self._engines: Dict[str, RapidOCR] = {}

    def model_for(
        self,
        doc_type: Optional[str] = None,
        filename: str = "",
        text: str = "",
    ) -> Path:
        """Resolve which checkpoint reads this document."""
        resolved = (doc_type or "").strip().upper()
        if not resolved:
            guess = classify_document(text or "", filename or "")
            if guess.get("confidence", 0) > 0:
                resolved = str(guess.get("type", "")).upper()
        if resolved in self.alt_types:
            return self.alt_model
        return self.default_model

    def _engine(self, model: Path) -> RapidOCR:
        key = str(model)
        if key not in self._engines:
            self._engines[key] = _build(model, self.dict_path, self.params)
        return self._engines[key]

    def __call__(
        self,
        img_or_path,
        doc_type: Optional[str] = None,
        filename: str = "",
        text: str = "",
        **kwargs,
    ):
        model = self.model_for(doc_type=doc_type, filename=filename, text=text)
        return self._engine(model)(img_or_path, **kwargs)
