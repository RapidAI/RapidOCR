# -*- encoding: utf-8 -*-
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


def test_ort_infer_session_release():
    """OrtInferSession.release() should delete the session and set it to None."""
    from rapidocr.inference_engine.onnxruntime.main import OrtInferSession

    obj = object.__new__(OrtInferSession)
    obj.session = MagicMock()

    obj.release()

    assert obj.session is None


def test_ort_infer_session_release_already_none():
    """Calling release() when session is already None should not raise."""
    from rapidocr.inference_engine.onnxruntime.main import OrtInferSession

    obj = object.__new__(OrtInferSession)
    obj.session = None

    obj.release()  # Should not raise

    assert obj.session is None


def test_text_detector_release():
    """TextDetector.release() should call session.release() and set session to None."""
    from rapidocr.ch_ppocr_det.main import TextDetector

    obj = object.__new__(TextDetector)
    mock_session = MagicMock()
    obj.session = mock_session

    obj.release()

    mock_session.release.assert_called_once()
    assert obj.session is None


def test_text_classifier_release():
    """TextClassifier.release() should call session.release() and set session to None."""
    from rapidocr.ch_ppocr_cls.main import TextClassifier

    obj = object.__new__(TextClassifier)
    mock_session = MagicMock()
    obj.session = mock_session

    obj.release()

    mock_session.release.assert_called_once()
    assert obj.session is None


def test_text_recognizer_release():
    """TextRecognizer.release() should call session.release() and set session to None."""
    from rapidocr.ch_ppocr_rec.main import TextRecognizer

    obj = object.__new__(TextRecognizer)
    mock_session = MagicMock()
    obj.session = mock_session

    obj.release()

    mock_session.release.assert_called_once()
    assert obj.session is None


def test_rapidocr_release():
    """RapidOCR.release() should release all three pipeline components."""
    from rapidocr.main import RapidOCR

    obj = object.__new__(RapidOCR)
    obj.text_det = MagicMock()
    obj.text_cls = MagicMock()
    obj.text_rec = MagicMock()

    obj.release()

    obj.text_det.release.assert_called_once()
    obj.text_cls.release.assert_called_once()
    obj.text_rec.release.assert_called_once()


def test_rapidocr_context_manager():
    """RapidOCR should work as a context manager and release on exit."""
    from rapidocr.main import RapidOCR

    obj = object.__new__(RapidOCR)
    obj.text_det = MagicMock()
    obj.text_cls = MagicMock()
    obj.text_rec = MagicMock()

    with obj as ocr:
        assert ocr is obj

    obj.text_det.release.assert_called_once()
    obj.text_cls.release.assert_called_once()
    obj.text_rec.release.assert_called_once()


def test_rapidocr_release_idempotent():
    """Calling release() multiple times should not raise."""
    from rapidocr.main import RapidOCR

    obj = object.__new__(RapidOCR)
    obj.text_det = MagicMock()
    obj.text_cls = MagicMock()
    obj.text_rec = MagicMock()

    obj.release()
    obj.release()  # Should not raise
