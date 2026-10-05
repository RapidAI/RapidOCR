"""Model download follows rapidocr 3.x: catalog, filenames, SHA256, and runtime fetch."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from rapidocr import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
from rapidocr.typings import TaskType
from rapidocr.inference_engine.base import FileInfo, InferSession
from rapidocr.utils.download_models import download_models
from rapidocr.utils.model_resolver import resolve_model_key
from rapidocr.utils.parse_parameters import ParseParams

_PACKAGE_CONFIG = Path(__file__).resolve().parents[1] / "src" / "rapidocr" / "config.yaml"


def _write_config(dest: Path, **global_overrides) -> Path:
    loaded = yaml.safe_load(_PACKAGE_CONFIG.read_text(encoding="utf-8"))
    loaded["Global"]["model_root_dir"] = str(dest)
    loaded["Global"]["use_cls"] = False
    loaded["Det"]["model_type"] = "tiny"
    loaded["Rec"]["model_type"] = "tiny"
    loaded["Global"].update(global_overrides)
    path = dest / "config.yaml"
    dest.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(loaded), encoding="utf-8")
    return path


def test_catalog_matches_main_v6_and_cls_keys():
    rec = InferSession.get_model_url(
        FileInfo(EngineType.ONNXRUNTIME, OCRVersion.PPOCRV6, TaskType.REC, LangRec.CH, ModelType.TINY)
    )
    assert rec["model_dir"].endswith("/onnx/PP-OCRv6/rec/PP-OCRv6_rec_tiny.onnx")
    assert rec["SHA256"] == "e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b"
    assert rec["dict_url"].endswith("/PP-OCRv6_rec_tiny/ppocrv6_tiny_dict.txt")

    det = InferSession.get_model_url(
        FileInfo(EngineType.PPOCR_CPP, OCRVersion.PPOCRV6, TaskType.DET, LangDet.CH, ModelType.SMALL)
    )
    assert det["model_dir"].endswith("/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx")
    assert det["SHA256"] == "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"

    cls = InferSession.get_model_url(
        FileInfo(EngineType.ONNXRUNTIME, OCRVersion.PPOCRV4, TaskType.CLS, LangCls.CH, ModelType.MOBILE)
    )
    assert cls["model_dir"].endswith("/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx")
    assert InferSession.get_dict_key_url(
        FileInfo(EngineType.ONNXRUNTIME, OCRVersion.PPOCRV6, TaskType.REC, "zh", ModelType.SMALL)
    ).endswith("/PP-OCRv6_rec_small/ppocrv6_dict.txt")


def test_tiny_rejects_japanese():
    with pytest.raises(ValueError, match="lang_type='japan'"):
        resolve_model_key(TaskType.REC, OCRVersion.PPOCRV6, "japan", ModelType.TINY)
    assert resolve_model_key(TaskType.REC, OCRVersion.PPOCRV6, "ja", ModelType.SMALL) == "multi_PP-OCRv6_rec_small"


def test_download_models_missing_and_empty(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        download_models(tmp_path / "missing.yaml")
    config = _write_config(tmp_path, use_det=False, use_cls=False, use_rec=False)
    with pytest.raises(ValueError, match="all false"):
        download_models(config)


def test_download_models_saves_url_filenames(tmp_path: Path, capsys):
    config = _write_config(tmp_path / "models", use_cls=True)
    assert download_models(config) is None
    printed = capsys.readouterr().out
    root = (tmp_path / "models").resolve()
    assert f"Models downloaded to {root}" in printed
    assert "Please initialize RapidOCR" in printed
    det = root / "PP-OCRv6_det_tiny.onnx"
    rec = root / "PP-OCRv6_rec_tiny.onnx"
    dictionary = root / "ppocrv6_tiny_dict.txt"
    cls = root / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    assert det.is_file() and det.stat().st_size > 0
    assert rec.is_file() and rec.stat().st_size > 0
    assert dictionary.is_file() and dictionary.stat().st_size > 0
    assert cls.is_file() and cls.stat().st_size > 0
    from rapidocr.utils.utils import get_file_sha256

    assert get_file_sha256(det) == "f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5"
    assert get_file_sha256(rec) == "e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b"
    assert get_file_sha256(cls) == "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"
    assert download_models(config) is None


def test_runtime_downloads_when_paths_are_null(tmp_path: Path):
    image = Path(os.environ.get("PPOCR_TEST_IMAGE", "/tmp/ppocr-models/hello.ppm"))
    if not image.is_file() or not os.environ.get("PPOCR_LIBRARY"):
        pytest.skip("set PPOCR_LIBRARY and PPOCR_TEST_IMAGE to run native OCR")
    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(tmp_path),
            "Global.use_cls": False,
            "Det.model_type": ModelType.TINY,
            "Rec.model_type": ModelType.TINY,
        }
    )
    assert engine.cfg.Global.model_root_dir == str(tmp_path) or Path(engine.cfg.Global.model_root_dir) == tmp_path
    result = engine(image)
    assert result.txts
    assert any("RapidOCR" in text or "Hello" in text for text in result.txts)
    assert (tmp_path / "PP-OCRv6_det_tiny.onnx").is_file()
    assert (tmp_path / "PP-OCRv6_rec_tiny.onnx").is_file()
    assert (tmp_path / "ppocrv6_tiny_dict.txt").is_file()
    assert not (tmp_path / "ch_ppocr_mobile_v2.0_cls_mobile.onnx").exists()


def test_default_model_root_is_package_models():
    cfg = ParseParams.load(None)
    assert cfg.Global.model_root_dir is None
    engine = RapidOCR(params={"Global.use_cls": False})
    assert Path(engine.cfg.Global.model_root_dir).name == "models"
