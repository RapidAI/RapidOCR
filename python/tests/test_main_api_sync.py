"""Python API changes ported from main (f65c7da..7cfd3d9, #755)."""

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidocr import LoadImageError, ModelType, OCRVersion, RapidOCR, RawMemoryImage
from rapidocr import main as rapidocr_main
from rapidocr.inference_engine.base import FileInfo, InferSession
from rapidocr.typings import EngineType, TaskType
from rapidocr.utils.load_image import LoadImage, MemoryImage
from rapidocr.utils.model_resolver import (
    MODEL_ROUTES,
    list_supported_langs,
    resolve_model_key,
    route_to_model_key,
)
from tools.validate_model_routes import validate_model_routes

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def native_engine():
    import os

    det = os.environ.get("PPOCR_TEST_DET", "/tmp/ppocr-models/det_tiny.onnx")
    rec = os.environ.get("PPOCR_TEST_REC", "/tmp/ppocr-models/rec_tiny.onnx")
    dictionary = os.environ.get("PPOCR_TEST_DICT", "/tmp/ppocr-models/dict.txt")
    if not all(Path(p).is_file() for p in (det, rec, dictionary)):
        pytest.skip("set PPOCR_TEST_DET/REC/DICT to run native OCR checks")
    try:
        return RapidOCR(params={"Det.model_path": det, "Rec.model_path": rec,
                                "Rec.rec_keys_path": dictionary, "Global.use_cls": False})
    except OSError as exc:
        pytest.skip(f"native library unavailable: {exc}")


# --- RawMemoryImage -------------------------------------------------------


def make_raw(img, pixel_format, stride=None, bottom_up=False):
    height, width = img.shape[:2]
    row_bytes = img.shape[1] * (1 if img.ndim == 2 else img.shape[2])
    stride = row_bytes if stride is None else stride
    rows = img[::-1] if bottom_up else img
    raw = bytearray(stride * height)
    for i, row in enumerate(rows):
        raw[i * stride : i * stride + row_bytes] = row.tobytes()
    buffer = ctypes.create_string_buffer(bytes(raw))
    return (
        RawMemoryImage(ctypes.addressof(buffer), len(raw), width, height,
                       pixel_format=pixel_format, stride=stride, bottom_up=bottom_up),
        buffer,
    )


BGR = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)


@pytest.mark.parametrize(
    "pixel_format,source",
    [
        ("BGR", BGR),
        ("RGB", np.array([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8)),
        ("BGRA", np.array([[[1, 2, 3, 0], [4, 5, 6, 255]]], dtype=np.uint8)),
        ("BGRX", np.array([[[1, 2, 3, 8], [4, 5, 6, 9]]], dtype=np.uint8)),
        ("RGBA", np.array([[[3, 2, 1, 0], [6, 5, 4, 255]]], dtype=np.uint8)),
        ("rgbx", np.array([[[3, 2, 1, 8], [6, 5, 4, 9]]], dtype=np.uint8)),
    ],
)
def test_raw_memory_image_matches_bgr_ndarray(pixel_format, source):
    memory_img, buffer = make_raw(source, pixel_format)
    assert np.array_equal(LoadImage().load_raw_memory_image(memory_img), BGR)
    # Same as passing the BGR ndarray (what main returns for this input).
    np.testing.assert_array_equal(LoadImage()(memory_img), LoadImage()(BGR))
    assert buffer is not None


def test_raw_memory_image_gray_stride_bottom_up():
    gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    memory_img, buffer = make_raw(gray, "GRAY8", stride=4, bottom_up=True)
    expected = np.stack([gray, gray, gray], axis=-1)
    np.testing.assert_array_equal(LoadImage()(memory_img), expected)
    assert buffer is not None


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"address": 0}, "invalid memory address"),
        ({"width": 0}, "width and height must be positive"),
        ({"pixel_format": "YUV"}, "unsupported pixel format"),
        ({"stride": 5}, "stride .* is smaller than row size"),
        ({"length": 8}, "buffer is too small"),
    ],
)
def test_raw_memory_image_rejects_invalid_metadata(kwargs, error):
    raw = bytes(12)
    buffer = ctypes.create_string_buffer(raw)
    params = {"address": ctypes.addressof(buffer), "length": len(raw), "width": 2,
              "height": 2, "pixel_format": "BGR", "stride": 6}
    params.update(kwargs)
    with pytest.raises(LoadImageError, match=error):
        LoadImage()(RawMemoryImage(**params))


def test_raw_memory_image_same_ocr_as_path(native_engine):
    path = DATA / "hello.png"
    rgb = np.asarray(Image.open(path).convert("RGB"))
    memory_img, buffer = make_raw(np.ascontiguousarray(rgb), "RGB")
    a = native_engine(path)
    b = native_engine(memory_img)
    assert a.txts == b.txts
    np.testing.assert_allclose(np.asarray(a.boxes), np.asarray(b.boxes))
    assert buffer is not None


# --- EXIF for bytes / PIL -------------------------------------------------


@pytest.mark.parametrize("input_type", ["bytes", "bytearray", "pil"])
@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_exif_input_types_match_path(tmp_path, input_type, orientation):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = orientation
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif)
    loader = LoadImage()
    rotations = {1: 0, 3: 2, 6: 3, 8: 1}
    expected = np.rot90(pixels, rotations[orientation])  # packed RGB
    np.testing.assert_array_equal(loader(path), expected)
    if input_type == "bytes":
        actual = loader(path.read_bytes())
    elif input_type == "bytearray":
        actual = loader(bytearray(path.read_bytes()))
    else:
        with Image.open(path) as pil_image:
            original = np.array(pil_image)
            actual = loader(pil_image)
            assert pil_image.getexif()[274] == orientation
            np.testing.assert_array_equal(np.array(pil_image), original)
    np.testing.assert_array_equal(actual, expected)


def test_memory_image_reads_exif(tmp_path):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "o.png"
    image.save(path, exif=exif)
    data = path.read_bytes()
    buffer = ctypes.create_string_buffer(data)
    out = LoadImage()(MemoryImage(ctypes.addressof(buffer), len(data)))
    np.testing.assert_array_equal(out, np.rot90(pixels, 3))


# --- model_routes in YAML -------------------------------------------------


def test_model_routes_schema_is_valid():
    assert "PP-OCRv6" in MODEL_ROUTES
    validate_model_routes(MODEL_ROUTES)


def test_model_routes_reject_unknown_enum_values():
    with pytest.raises(ValueError, match="Invalid OCR version"):
        validate_model_routes({"PP-OCRv7": {}})
    with pytest.raises(ValueError, match="Invalid task"):
        validate_model_routes({"PP-OCRv6": {"recognize": {}}})
    with pytest.raises(ValueError, match="Invalid model type"):
        validate_model_routes({"PP-OCRv6": {"rec": {"large": []}}})


def test_every_route_resolves_in_the_onnx_registry():
    registry = InferSession.model_info["onnxruntime"]
    for version, version_cfg in MODEL_ROUTES.items():
        if version == "_defs":
            continue
        for task, task_cfg in version_cfg.items():
            if task == "aliases":
                continue
            for model_type, routes in task_cfg.items():
                if model_type == "aliases":
                    continue
                for route in routes:
                    key = route["model_key"]
                    assert key in (registry.get(version, {}).get(task) or {}), key
                    for lang in route["supported_langs"]:
                        assert route_to_model_key(TaskType(task), OCRVersion(version),
                                                  lang, ModelType(model_type)) == key
                    info = InferSession.get_model_url(FileInfo(
                        EngineType.ONNXRUNTIME, OCRVersion(version), TaskType(task),
                        route["supported_langs"][0], ModelType(model_type)))
                    assert info == registry[version][task][key]


def test_ppocrv6_routes():
    assert route_to_model_key(TaskType.DET, OCRVersion.PPOCRV6, "fr", ModelType.TINY) == "multi_PP-OCRv6_det_tiny"
    assert route_to_model_key(TaskType.REC, OCRVersion.PPOCRV6, "ja", ModelType.SMALL) == "multi_PP-OCRv6_rec_small"
    assert resolve_model_key is route_to_model_key
    with pytest.raises(ValueError, match="japan.*PP-OCRv6 tiny"):
        route_to_model_key(TaskType.REC, OCRVersion.PPOCRV6, "ja", ModelType.TINY)


def test_aliases_are_scoped_to_routes():
    assert route_to_model_key(TaskType.REC, OCRVersion.PPOCRV4, "ja", ModelType.MOBILE) == "japan_PP-OCRv4_rec_mobile"
    assert route_to_model_key(TaskType.CLS, OCRVersion.PPOCRV4, "ch", ModelType.MOBILE) == "ch_ppocr_mobile_v2.0_cls_mobile"


def test_list_supported_langs():
    tiny = list_supported_langs(TaskType.REC, OCRVersion.PPOCRV6, ModelType.TINY)
    assert "japan" not in tiny and "fr" in tiny
    assert "japan" in list_supported_langs(TaskType.REC, OCRVersion.PPOCRV6)
    assert list_supported_langs(TaskType.CLS, OCRVersion.PPOCRV5, ModelType.SERVER) == ["ch", "multi"]
    assert RapidOCR.list_supported_langs("rec", "PP-OCRv6", "tiny") == tiny
    assert RapidOCR.list_supported_langs(TaskType.REC, OCRVersion.PPOCRV6, ModelType.SMALL) == \
        list_supported_langs(TaskType.REC, OCRVersion.PPOCRV6, ModelType.SMALL)


def test_cli_list_langs(capsys):
    rapidocr_main.main(["--list-langs", "--task", "rec", "--ocr_version", "PP-OCRv6", "--model_type", "tiny"])
    out = capsys.readouterr().out.split()
    assert out == list_supported_langs(TaskType.REC, OCRVersion.PPOCRV6, ModelType.TINY)
    rapidocr_main.main(["--list-langs", "--task", "det"])
    assert "japan" in capsys.readouterr().out.split()


def test_cli_accepts_img_path_flag():
    args = rapidocr_main.parse_args(["-img", "a.png", "--lang_type", "en", "--text_score", "0.6"])
    assert args.img_path == "a.png" and args.lang_type == "en" and args.text_score == 0.6


# --- model_root_dir ------------------------------------------------------


@pytest.mark.parametrize("as_type", [Path, str])
def test_rec_dict_downloaded_into_model_root_dir(tmp_path, monkeypatch, as_type):
    import rapidocr.utils.download_file as download_file
    import rapidocr.utils.download_models as download_models

    saved = []
    monkeypatch.setattr(download_file.DownloadFile, "run",
                        staticmethod(lambda params: saved.append(Path(params.save_path))))
    monkeypatch.setattr(download_models, "download_task", lambda root, info: None)
    cfg = rapidocr_main._load_config(None, {"Global.model_root_dir": as_type(tmp_path),
                                             "Global.use_cls": False})
    det, rec, dictionary = rapidocr_main._resolve_models(cfg)
    for p in (det, rec, dictionary):
        assert Path(p).parent == tmp_path.resolve()
    assert saved == [Path(dictionary)]


    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert "opencv" not in text
    import rapidocr.utils.load_image as li
    assert "cv2" not in li.__dict__
