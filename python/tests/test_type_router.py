
import pytest

from rapidocr.utils.type_router import ALT_MODEL_TYPES, DEFAULT_MODEL, TypeModelRouter


@pytest.fixture()
def router():
    return TypeModelRouter()


def test_explicit_type_uses_alt_model(router):
    assert router.model_for(doc_type="VOUCHER").name == "spanish_rec_v19.onnx"
    assert router.model_for(doc_type="familiar").name == "spanish_rec_v19.onnx"


def test_known_type_that_alt_model_lost_stays_default(router):
    # CV, certificates and blueprints measured better on the default checkpoint.
    for doc_type in ("CV", "CERTIFICADO", "PLANO", "ACTA", "DNI", "ESSALUD"):
        assert router.model_for(doc_type=doc_type).name == DEFAULT_MODEL.name


def test_unknown_type_falls_back_to_default(router):
    assert router.model_for(doc_type="NO_EXISTE").name == DEFAULT_MODEL.name


def test_filename_hint_classifies_without_explicit_type(router):
    assert router.model_for(filename="06. VOUCHER.pdf").name == "spanish_rec_v19.onnx"


def test_text_hint_classifies_without_explicit_type(router):
    text = "voucher constancia de abono numero de operacion cuenta de ahorros"
    assert router.model_for(text=text).name == "spanish_rec_v19.onnx"


def test_alt_types_document_the_measured_choice():
    # Each entry must carry the CER comparison that justifies routing there.
    for doc_type, evidence in ALT_MODEL_TYPES.items():
        assert "vs" in evidence
