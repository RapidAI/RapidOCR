import os
from pathlib import Path

import numpy as np
import pytest

from rapidocr import EngineType, LoadImageError, RapidOCR
from rapidocr.output import RapidOCROutput, filter_by_score


def test_filter_and_json():
    boxes, txts, scores = filter_by_score(
        [[[0, 0], [10, 0], [10, 5], [0, 5]], [[1, 1], [2, 1], [2, 2], [1, 2]]],
        ["Hello", "  "],
        [0.9, 0.99],
        0.5,
    )
    assert txts == ("Hello",)
    result = RapidOCROutput(boxes=boxes, txts=txts, scores=scores, elapse_list=[None, None, 0.2])
    assert len(result) == 1
    assert result.elapse == pytest.approx(0.2)
    payload = result.to_json()
    assert payload[0]["txt"] == "Hello"
    assert "Hello" in result.to_markdown()


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        RapidOCR(params={"NotASection.thresh": 0.2})


def test_unsupported_engine():
    with pytest.raises(NotImplementedError):
        RapidOCR(params={"Det.engine_type": EngineType.OPENVINO, "Rec.engine_type": EngineType.OPENVINO})


def test_bad_image_type():
    engine = RapidOCR(params={"Det.model_path": "a", "Rec.model_path": "b", "Rec.rec_keys_path": "c"})
    with pytest.raises(LoadImageError):
        engine(12345)


def test_cpu_info_and_ocr():
    det = os.environ.get("PPOCR_TEST_DET")
    rec = os.environ.get("PPOCR_TEST_REC")
    dictionary = os.environ.get("PPOCR_TEST_DICT")
    image = os.environ.get("PPOCR_TEST_IMAGE")
    if not all((det, rec, dictionary, image, os.environ.get("PPOCR_LIBRARY"))):
        pytest.skip("set PPOCR_LIBRARY and PPOCR_TEST_* to run native OCR")
    from rapidocr import cpu_info

    info = cpu_info()
    assert info["active_isa"] in {"scalar", "avx2", "avx512", "neon"}
    if info["avx512"]:
        assert info["avx2"]
    engine = RapidOCR(
        params={
            "Det.model_path": det,
            "Rec.model_path": rec,
            "Rec.rec_keys_path": dictionary,
            "Global.use_cls": False,
            "Global.text_score": 0.3,
        }
    )
    result = engine(image)
    assert isinstance(result, RapidOCROutput)
    assert result.elapse >= 0
    array = np.asarray(result.img)
    again = engine(array)
    assert isinstance(again, RapidOCROutput)
    assert Path(image).is_file()
