"""Public rapidocr 3.x usage, run against rapidocr 5.

The cases follow ``python/demo.py`` and the public tests on ``main``
(``test_input.py``, ``test_to_other_format.py``, ``test_vis.py``,
``test_default_config.py``): construction, ``__call__`` arguments, output
fields, and input types. Items the native engine cannot match are asserted
as the 5.x behavior and listed in ``REMAINING_INCOMPATIBILITIES``.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rapidocr import (
    EngineType,
    LangCls,
    LangDet,
    LangRec,
    LoadImageError,
    ModelType,
    OCRVersion,
    RapidOCR,
    RapidOCRError,
    VisRes,
)
from rapidocr.typings import TaskType
from rapidocr.utils.load_image import LoadImage, MemoryImage
from rapidocr.utils.parse_parameters import ParseParams

REMAINING_INCOMPATIBILITIES = (
    "use_cls is accepted and ignored; there is no angle classifier",
    "use_det=False or use_rec=False raises RapidOCRError (no det-only or rec-only)",
    "return_word_box and return_single_char_box do not produce word boxes",
    "PP-OCRv4 and PP-OCRv5 model versions raise NotImplementedError",
    "EngineType other than ONNXRUNTIME and PPOCR_CPP raises NotImplementedError",
    "EngineConfig is stored and not applied (no CUDA, OpenVINO, Paddle, Torch, TensorRT, MNN)",
    "elapse_list is [None, None, total], not separate det/cls/rec times",
    "vis() returns Pillow RGB, not OpenCV BGR",
    "detector normalization is ImageNet mean/std, not the 3.x mean/std of 0.5",
    "only the PP-OCRv6 ch det/rec bundles are downloaded; other languages are not",
)


def test_remaining_incompatibilities_are_listed():
    assert len(REMAINING_INCOMPATIBILITIES) >= 8


def test_construct_default_matches_main_public_keys():
    cfg = ParseParams.load(None)
    assert cfg.Global.text_score == 0.5
    assert cfg.Global.use_det is True
    assert cfg.Global.use_cls is True
    assert cfg.Global.use_rec is True
    assert cfg.Global.min_height == 30
    assert cfg.Global.width_height_ratio == 8
    assert cfg.Global.max_side_len == 2000
    assert cfg.Global.min_side_len == 30
    assert cfg.Global.return_word_box is False
    assert cfg.Global.return_single_char_box is False
    assert cfg.Det.engine_type == EngineType.ONNXRUNTIME
    assert cfg.Det.lang_type == LangDet.CH
    assert cfg.Det.model_type == ModelType.SMALL
    assert cfg.Det.ocr_version == OCRVersion.PPOCRV6
    assert cfg.Det.task_type == TaskType.DET
    assert cfg.Det.limit_side_len == 736
    assert cfg.Det.thresh == 0.3
    assert cfg.Det.box_thresh == 0.5
    assert cfg.Det.unclip_ratio == 1.6
    assert cfg.Cls.ocr_version == OCRVersion.PPOCRV4
    assert cfg.Cls.lang_type == LangCls.CH
    assert cfg.Cls.model_type == ModelType.MOBILE
    assert cfg.Cls.task_type == TaskType.CLS
    assert cfg.Rec.ocr_version == OCRVersion.PPOCRV6
    assert cfg.Rec.lang_type == LangRec.CH
    assert cfg.Rec.model_type == ModelType.SMALL
    assert cfg.Rec.task_type == TaskType.REC
    assert cfg.Rec.rec_batch_num == 6
    assert cfg.EngineConfig.onnxruntime.use_cuda is False
    assert cfg.EngineConfig.mnn == {}


def test_construct_with_params_and_update_params():
    engine = RapidOCR(
        params={
            "Det.model_type": ModelType.TINY,
            "Rec.model_type": ModelType.TINY,
            "Global.use_cls": False,
            "Global.text_score": 0.3,
        }
    )
    assert engine.cfg.Det.model_type == ModelType.TINY
    assert engine.use_cls is False
    assert engine.text_det is None
    engine.update_params(box_thresh=0.7, unclip_ratio=2.0)
    assert engine.cfg.Det.box_thresh == 0.7
    assert engine.cfg.Det.unclip_ratio == 2.0
    engine.text_det = type("Det", (), {"postprocess_op": type("Post", (), {"box_thresh": 0.5, "unclip_ratio": 1.6})()})()
    engine.update_params(box_thresh=0.7, unclip_ratio=2.0)
    assert engine.text_det.postprocess_op.box_thresh == 0.7


@pytest.mark.parametrize("params", [{"text_score": 0.5}, {"Global": 0.5}, {"Unknown.key": 1}])
def test_invalid_init_parameters(params):
    cfg = ParseParams.load(None)
    with pytest.raises(ValueError):
        ParseParams.update_batch(cfg, params)


def test_nested_engine_config_is_stored():
    cfg = ParseParams.load(None)
    ParseParams.update_batch(cfg, {"EngineConfig.onnxruntime.use_cuda": True})
    assert cfg.EngineConfig.onnxruntime.use_cuda is True


def test_demo_shape_vis_class_is_exported():
    assert VisRes is not None
    blank = np.zeros((16, 16, 3), dtype=np.uint8)
    box = np.array([[[1, 1], [10, 1], [10, 8], [1, 8]]], dtype=np.float32)
    painted = VisRes()(blank, box, ["hi"], [0.9])
    assert painted.shape == (16, 16, 3)


def test_load_image_types_and_memory_image(tmp_path: Path):
    image = Image.new("RGB", (24, 12), (255, 255, 255))
    path = tmp_path / "sample.png"
    image.save(path)
    loader = LoadImage()
    from_path = loader(path)
    from_str = loader(str(path))
    encoded = path.read_bytes()
    from_bytes = loader(encoded)
    from_pil = loader(image)
    assert from_path.shape == from_bytes.shape == from_pil.shape == (12, 24, 3)
    np.testing.assert_array_equal(from_path, from_str)
    buffer = ctypes.create_string_buffer(encoded)
    from_memory = loader(MemoryImage(ctypes.addressof(buffer), len(encoded)))
    np.testing.assert_array_equal(from_memory, from_bytes)
    bgr = from_path[:, :, ::-1]
    swapped = loader(bgr)
    np.testing.assert_array_equal(swapped, from_path)
    gray = from_path[:, :, :1]
    assert loader(gray).shape[2] == 3
    two = np.dstack([from_path[:, :, 0], np.full(from_path.shape[:2], 255, np.uint8)])
    assert loader(two).shape[2] == 3
    rgba = np.dstack([from_path, np.full(from_path.shape[:2], 255, np.uint8)])
    assert loader(rgba).shape[2] == 3
    mode_one = Image.new("1", (8, 8), 1)
    assert loader(mode_one).shape == (8, 8, 3)


def test_load_image_from_http_url(tmp_path: Path):
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    import threading

    image = Image.new("RGB", (8, 6), (10, 20, 30))
    path = tmp_path / "net.png"
    image.save(path)
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        loaded = LoadImage()(f"http://127.0.0.1:{port}/net.png")
        assert loaded.shape == (6, 8, 3)
        np.testing.assert_array_equal(loaded[0, 0], [10, 20, 30])
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_memory_image_invalid():
    loader = LoadImage()
    with pytest.raises(LoadImageError, match="MemoryImage loading failed"):
        loader(MemoryImage(0, 1))
    encoded = b"not an image"
    buffer = ctypes.create_string_buffer(encoded)
    with pytest.raises(LoadImageError, match="MemoryImage loading failed"):
        loader(MemoryImage(ctypes.addressof(buffer), len(encoded)))


def test_empty_and_bad_input_match_main_shape():
    engine = RapidOCR(params=_local_or_placeholder_params())
    with pytest.raises(LoadImageError):
        engine(None)
    if not _models_ready():
        return
    result = engine(np.zeros((64, 64, 3), dtype=np.uint8))
    assert result.boxes is None
    assert result.txts is None
    assert result.img is None
    assert len(result) == 0


def test_gap_det_only_and_rec_only():
    engine = RapidOCR(params=_local_or_placeholder_params())
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(RapidOCRError):
        engine(blank, use_det=True, use_cls=False, use_rec=False)
    with pytest.raises(RapidOCRError):
        engine(blank, use_det=False, use_cls=False, use_rec=True)


def test_gap_ppocr_v4_and_openvino():
    with pytest.raises(NotImplementedError):
        RapidOCR(
            params={
                "Det.ocr_version": OCRVersion.PPOCRV4,
                "Rec.ocr_version": OCRVersion.PPOCRV4,
            }
        )
    with pytest.raises(NotImplementedError):
        RapidOCR(params={"Det.engine_type": EngineType.OPENVINO, "Rec.engine_type": EngineType.OPENVINO})


def test_call_output_fields_on_sample(tmp_path: Path):
    image = _sample_image()
    if image is None or not _models_ready():
        pytest.skip("set PPOCR_TEST_DET/REC/DICT/IMAGE or place the tiny bundle and hello.ppm")
    engine = RapidOCR(params=_model_params())
    result = engine(str(image))
    again = engine(image.read_bytes())
    assert result.txts == again.txts
    assert result.txts
    if image.name == "hello.ppm":
        assert result.txts == ("Hello RapidOCR 123",)
    assert result.scores is not None and len(result.scores) == len(result.txts)
    assert result.boxes is not None and result.boxes.shape[1:] == (4, 2)
    assert result.elapse >= 0
    assert result.elapse_list[0] is None and result.elapse_list[1] is None
    payload = result.to_json()
    assert payload[0]["txt"] == result.txts[0]
    assert "box" in payload[0] and "score" in payload[0]
    markdown = result.to_markdown()
    assert result.txts[0] in markdown
    vis = result.vis(str(tmp_path / "vis.png"))
    assert vis is not None and vis.shape[2] == 3
    assert (tmp_path / "vis.png").is_file()
    pil = Image.open(image).convert("RGB")
    from_pil = engine(pil)
    assert from_pil.txts == result.txts
    array = np.asarray(pil)[:, :, ::-1]
    from_array = engine(array)
    assert from_array.txts == result.txts
    with_cls = engine(image, use_cls=True)
    assert with_cls.txts == result.txts
    warned = engine(image, return_word_box=True)
    assert warned.txts == result.txts
    assert warned.word_results == ()
    blocked = engine(image, text_score=1.0)
    assert blocked.boxes is None


def _local_or_placeholder_params():
    if _models_ready():
        return _model_params()
    return {
        "Det.model_path": "det.onnx",
        "Rec.model_path": "rec.onnx",
        "Rec.rec_keys_path": "dict.txt",
        "Global.use_cls": False,
    }


def _models_ready() -> bool:
    params = _model_params()
    return all(Path(params[key]).is_file() for key in ("Det.model_path", "Rec.model_path", "Rec.rec_keys_path"))


def _model_params():
    det = os.environ.get("PPOCR_TEST_DET", "/tmp/ppocr-models/det_tiny.onnx")
    rec = os.environ.get("PPOCR_TEST_REC", "/tmp/ppocr-models/rec_tiny.onnx")
    dictionary = os.environ.get("PPOCR_TEST_DICT", "/tmp/ppocr-models/dict.txt")
    return {
        "Det.model_path": det,
        "Rec.model_path": rec,
        "Rec.rec_keys_path": dictionary,
        "Det.model_type": "tiny",
        "Rec.model_type": "tiny",
        "Global.use_cls": False,
        "Global.text_score": 0.3,
    }


def _sample_image():
    configured = os.environ.get("PPOCR_TEST_IMAGE", "/tmp/ppocr-models/hello.ppm")
    path = Path(configured)
    return path if path.is_file() else None
