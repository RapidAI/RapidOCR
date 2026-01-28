# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

from pytest import mark

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR

tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"


@mark.parametrize(
    "engine_type",
    [
        EngineType.ONNXRUNTIME,
        EngineType.PADDLE,
        EngineType.OPENVINO,
        EngineType.TORCH,
        EngineType.MNN,
    ],
)
def test_ppocrv5_rec_mobile(engine_type):
    engine = RapidOCR(
        params={
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.engine_type": engine_type,
        }
    )
    img_path = tests_dir / "text_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)

    assert result.txts is not None
    assert result.txts[0] == "韩国小馆"


@mark.parametrize(
    "engine_type",
    [
        EngineType.ONNXRUNTIME,
        EngineType.PADDLE,
        EngineType.OPENVINO,
        EngineType.TORCH,
        EngineType.MNN,
    ],
)
def test_ppocrv5_det_mobile(engine_type):
    engine = RapidOCR(
        params={
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Det.engine_type": engine_type,
        }
    )
    img_path = tests_dir / "ch_en_num.jpg"
    result = engine(img_path, use_det=True, use_cls=False, use_rec=False)

    assert result.boxes is not None
    assert len(result.boxes) == 17


@mark.skipif(sys.platform.startswith("darwin"), reason="does not run on macOS")
def test_engine_openvino():
    engine = RapidOCR(
        params={
            "Det.engine_type": EngineType.OPENVINO,
            "Cls.engine_type": EngineType.OPENVINO,
            "Rec.engine_type": EngineType.OPENVINO,
        }
    )

    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == "正品促销"


@mark.parametrize(
    "ocr_version,gt",
    [(OCRVersion.PPOCRV4, "正品促销"), (OCRVersion.PPOCRV5, "大桶装更划算")],
)
def test_engine_paddle(ocr_version, gt):
    engine = RapidOCR(
        params={
            "Det.engine_type": EngineType.PADDLE,
            "Det.ocr_version": ocr_version,
            "Cls.engine_type": EngineType.PADDLE,
            "Rec.engine_type": EngineType.PADDLE,
        }
    )

    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == gt


def test_engine_torch():
    engine = RapidOCR(
        params={
            "Det.engine_type": EngineType.TORCH,
            "Cls.engine_type": EngineType.TORCH,
            "Rec.engine_type": EngineType.TORCH,
        }
    )

    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == "正品促销"
