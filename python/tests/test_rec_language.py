# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import pytest
from pytest import mark

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import (
    EngineType,
    LangRec,
    ModelType,
    OCRVersion,
    RapidOCR,
)

tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"


@pytest.fixture()
def engine():
    engine = RapidOCR()
    return engine


@mark.parametrize(
    "lang_type, img_path, gt",
    [
        (LangRec.CYRILLIC, "cyrillic.png", "Време је лепо данас."),
        (LangRec.DEVANAGARI, "devanagari_rec.png", "आज मौसम अच्छा है।"),
        (LangRec.TA, "ta.png", "இன்னைக்கு வானிலை ராம்ப நல்லா இருக்கு."),
        (LangRec.TE, "te.png", "ఈరోజ వాతావరణం చాలా బాగుంది."),
    ],
)
def test_multi_lang(lang_type, img_path, gt):
    engine = RapidOCR(
        params={
            "Rec.lang_type": lang_type,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
        }
    )
    img_path = tests_dir / img_path
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)

    assert result.txts is not None
    assert result.txts[0] == gt


def test_cyrillic_lang():
    engine = RapidOCR(
        params={
            "Rec.lang_type": LangRec.CYRILLIC,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
        }
    )
    img_path = tests_dir / "cyrillic.png"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)

    assert result.txts is not None
    assert result.txts[0] == "Време је лепо данас."


@mark.parametrize(
    "engine_type",
    [EngineType.ONNXRUNTIME, EngineType.OPENVINO, EngineType.PADDLE],
)
def test_arabic_lang(engine_type):
    engine = RapidOCR(
        params={
            "Rec.lang_type": LangRec.ARABIC,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": engine_type,
        }
    )
    img_path = tests_dir / "arabic.png"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)

    assert result.txts is not None
    assert result.txts[0] == "امروز هوا خوبه"


@mark.parametrize(
    "engine_type",
    [EngineType.ONNXRUNTIME, EngineType.OPENVINO, EngineType.PADDLE],
)
def test_el_lang(engine_type):
    engine = RapidOCR(
        params={
            "Rec.lang_type": LangRec.EL,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": engine_type,
        }
    )
    img_path = tests_dir / "el_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)

    assert result.txts is not None
    assert result.txts[0] == "Ωραίος καιρός σήμερα."


@mark.parametrize(
    "engine_type",
    [EngineType.ONNXRUNTIME, EngineType.OPENVINO, EngineType.PADDLE],
)
def test_th_lang(engine_type):
    engine = RapidOCR(
        params={
            "Rec.lang_type": LangRec.TH,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": engine_type,
        }
    )
    img_path = tests_dir / "th_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)
    assert result.txts is not None
    assert (
        result.txts[0] == "การแพทย์แผนจีนช่วยให้เด็กสมองพิการในคาซัคสถานกลับมาเดินได้"
    )


@mark.parametrize(
    "ocr_version",
    [OCRVersion.PPOCRV4, OCRVersion.PPOCRV5],
)
def test_en_lang(ocr_version):
    engine = RapidOCR(
        params={
            "Rec.lang_type": LangRec.EN,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": ocr_version,
        }
    )
    img_path = tests_dir / "en_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)

    assert result.txts is not None
    assert (
        result.txts[0]
        == "To facilitate the shot type analysis in videos, we collect MovieShots, a large-scale"
    )


def test_ch_doc_server():
    engine = RapidOCR(
        params={"Rec.lang_type": LangRec.CH_DOC, "Rec.model_type": ModelType.SERVER}
    )

    img_path = tests_dir / "ch_doc_server.png"
    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == "嫖娼"


def test_server_rec():
    engine = RapidOCR(
        params={"Rec.lang_type": LangRec.CH, "Rec.model_type": ModelType.SERVER}
    )
    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == "正品促销"


@mark.parametrize(
    "ocr_version,gt",
    [(OCRVersion.PPOCRV4, "베이징차오양"), (OCRVersion.PPOCRV5, "베이징 차오양,")],
)
def test_korean_lang(ocr_version, gt):
    engine = RapidOCR(
        params={
            "Rec.lang_type": LangRec.KOREAN,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": ocr_version,
        }
    )
    img_path = tests_dir / "korean.jpg"
    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == gt


@mark.parametrize(
    "engine_type",
    [EngineType.ONNXRUNTIME, EngineType.OPENVINO, EngineType.PADDLE],
)
def test_latin_lang(engine_type):
    engine = RapidOCR(
        params={
            "Rec.lang_type": LangRec.LATIN,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": engine_type,
        }
    )
    img_path = tests_dir / "latin.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)
    assert result.txts is not None
    assert (
        result.txts[0]
        == "Alphabetum in mundo hodie frequentissie adhibitum est alphabetum Latinum."
    )


@mark.parametrize(
    "engine_type",
    [EngineType.ONNXRUNTIME, EngineType.OPENVINO, EngineType.PADDLE],
)
def test_eslav_lang(engine_type):
    engine = RapidOCR(
        params={
            "Rec.lang_type": LangRec.ESLAV,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": engine_type,
        }
    )
    img_path = tests_dir / "eslav.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)
    assert result.txts is not None
    assert result.txts[0] == "Славянские языки — большая языковая семья."
