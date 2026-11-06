"""Document-type classifier for Spanish multi-type OCR post-processing.

Port of a production content-classifier, adapted to run inside RapidOCR so the
recognition pipeline can pick type-specific lexicons and regex guards (see the
multi-type Spanish strategy document, sections 2 and 4.6).

The taxonomy is intentionally data-driven: callers may override it (e.g. from a
backend quality-rules store) by passing ``taxonomy=`` to :func:`classify`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DEFAULT_TAXONOMY: Dict[str, Any] = {
    "document_types": [
        "CV",
        "CUL",
        "DNI",
        "FOTO",
        "CARNET_RETCC",
        "LICENCIA_CONDUCIR",
        "RECORD_CONDUCTOR",
        "ESSALUD",
        "VIDA_LEY",
        "VOUCHER",
        "DOCUMENTOS_FAMILIARES",
        "CERTIFICADO",
        "CONTRATO",
        "CONSTANCIA",
        "BLUEPRINT",
        "ESPEC_TECNICA",
        "PRESUPUESTO",
        "INFORME_OBRA",
        "CONTRATO_OBRA",
        "FACTURA",
        "RECIBO",
        "FORMULARIO",
        "TABLA",
        "INFORME",
        "OTRO",
    ],
    "filename_hints": {
        "cv": "CV",
        "curriculum": "CV",
        "resume": "CV",
        "cul": "CUL",
        "laboral": "CUL",
        "dni": "DNI",
        "documento": "DNI",
        "certificado": "CERTIFICADO",
        "cert": "CERTIFICADO",
        "contrato": "CONTRATO",
        "constancia": "CONSTANCIA",
        "blueprint": "BLUEPRINT",
        "plano": "BLUEPRINT",
        "factura": "FACTURA",
        "recibo": "RECIBO",
        "formulario": "FORMULARIO",
        "tabla": "TABLA",
        "presupuesto": "PRESUPUESTO",
        "informe": "INFORME",
        "voucher": "VOUCHER",
        "cuenta bancaria": "VOUCHER",
        "essalud": "ESSALUD",
        "vida ley": "VIDA_LEY",
        "carnet": "CARNET_RETCC",
        "retcc": "CARNET_RETCC",
        "licencia": "LICENCIA_CONDUCIR",
        "record de conductor": "RECORD_CONDUCTOR",
        "familiar": "DOCUMENTOS_FAMILIARES",
    },
    "keywords": {
        "CV": [
            "experiencia laboral",
            "educación",
            "habilidades",
            "referencias",
            "objetivo profesional",
            "perfil profesional",
            "formación académica",
        ],
        "CUL": [
            "certificado único laboral",
            "ministerio de trabajo",
            "mtpe",
            "registro de trabajadores",
            "historial laboral",
        ],
        "DNI": [
            "documento nacional",
            "dni",
            "reniec",
            "fecha de nacimiento",
            "lugar de nacimiento",
            "estado civil",
        ],
        "ESSALUD": [
            "seguro social de salud",
            "essalud",
            "prestaciones economicas",
            "prestaciones económicas",
            "acreditación de asegurado",
            "acreditacion de asegurado",
            "derechohabiente",
        ],
        "VOUCHER": [
            "voucher",
            "constancia de abono",
            "número de operación",
            "numero de operacion",
            "cuenta de ahorros",
            "importe abonado",
            "bcp",
            "interbank",
            "banco de credito",
            "banco de crédito",
        ],
        "VIDA_LEY": [
            "seguro de vida ley",
            "vida ley",
            "póliza de seguro",
            "poliza de seguro",
            "beneficiarios",
            "asegurado titular",
        ],
        "DOCUMENTOS_FAMILIARES": [
            "partida de nacimiento",
            "acta de matrimonio",
            "certificado de matrimonio",
            "partida de matrimonio",
            "declaración jurada de unión de hecho",
            "certificado de nacido vivo",
            "documento familiar",
        ],
        "RECORD_CONDUCTOR": [
            "récord de conductor",
            "record de conductor",
            "historial de infracciones",
            "licencia de conducir",
        ],
        "CERTIFICADO": [
            "certificamos",
            "se certifica",
            "hace constar",
            "otorga el presente",
            "capacitación",
            "curso",
            "diploma",
        ],
        "CONTRATO": [
            "contrato de trabajo",
            "empleador",
            "trabajador",
            "remuneración",
            "jornada laboral",
            "cláusula",
            "vigencia",
        ],
        "CONSTANCIA": [
            "constancia",
            "trabaja actualmente",
            "labora desde",
            "cargo actual",
        ],
        "BLUEPRINT": [
            "plano",
            "escala",
            "cota",
            "blueprint",
            "proyecto",
            "arquitectura",
            "estructura",
        ],
        "FACTURA": [
            "factura",
            "ruc",
            "igv",
            "subtotal",
            "total a pagar",
            "sunat",
        ],
        "RECIBO": [
            "recibo por honorarios",
            "recibo",
            "monto",
        ],
        "FORMULARIO": [
            "formulario",
            "solicitud",
        ],
        "PRESUPUESTO": [
            "presupuesto",
            "partida",
            "metrado",
            "cantidad",
            "precio unitario",
        ],
        "TABLA": [
            "tabla",
        ],
        "INFORME": [
            "informe",
            "resumen",
        ],
    },
    "weights": {"by_type": {"CUL": 15}, "default_keyword_weight": 10},
    "filename_hint_weight": 50,
    "minimum_confidence": 20,
    "maximum_confidence": 100,
    "fallback_type": "OTRO",
}


def classify(
    text: str,
    filename: str = "",
    taxonomy: Optional[Dict[str, Any]] = None,
    visual_hints: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Classify a document by text content + filename (+ optional visual hints).

    Returns ``{"type", "confidence", "scores"}``.
    """
    rules = taxonomy if taxonomy is not None else DEFAULT_TAXONOMY
    required = {
        "document_types",
        "filename_hints",
        "keywords",
        "weights",
        "filename_hint_weight",
        "minimum_confidence",
        "maximum_confidence",
        "fallback_type",
    }
    if not isinstance(rules, dict) or not required.issubset(rules):
        raise ValueError("taxonomy missing required keys")

    document_types: List[str] = rules["document_types"]
    filename_hints: Dict[str, str] = rules["filename_hints"]
    keywords: Dict[str, List[str]] = rules["keywords"]
    weights: Dict[str, Any] = rules["weights"]
    filename_hint_weight: int = rules["filename_hint_weight"]
    minimum_confidence: int = rules["minimum_confidence"]
    maximum_confidence: int = rules["maximum_confidence"]
    fallback_type: str = rules["fallback_type"]

    scores = {dt: 0 for dt in document_types}

    text_lower = (text or "").lower()
    filename_lower = (filename or "").lower()

    for hint, doc_type in filename_hints.items():
        if hint in filename_lower and doc_type in scores:
            scores[doc_type] += filename_hint_weight

    for doc_type, terms in keywords.items():
        if doc_type not in scores:
            continue
        w = weights.get("by_type", {}).get(doc_type, weights.get("default_keyword_weight", 10))
        for kw in terms:
            if kw in text_lower:
                scores[doc_type] += w

    vh = visual_hints or {}
    if vh.get("has_cad_lines"):
        scores["BLUEPRINT"] = scores.get("BLUEPRINT", 0) + filename_hint_weight
    if vh.get("has_dense_table"):
        for t in ("TABLA", "PRESUPUESTO", "FACTURA"):
            if t in scores:
                scores[t] = scores.get(t, 0) + 10

    best_type = max(scores, key=scores.get) if scores else fallback_type
    confidence = min(scores.get(best_type, 0), maximum_confidence)
    if confidence < minimum_confidence:
        best_type = fallback_type
        confidence = 0

    return {
        "type": best_type,
        "confidence": confidence,
        "scores": {k: v for k, v in scores.items() if v > 0},
    }