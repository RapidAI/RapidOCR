"""ctypes binding over the stable C ABI in ``include/ppocr/ppocr.h``."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

import numpy as np

PPOCR_OK = 0
PPOCR_BACKEND_CPU = 0
PPOCR_BACKEND_GPU = 1
PPOCR_BACKEND_HYBRID = 2


class Options(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_int),
        ("backend", ctypes.c_int),
        ("det_threshold", ctypes.c_float),
        ("det_box_threshold", ctypes.c_float),
        ("det_unclip_ratio", ctypes.c_float),
        ("det_limit_side_len", ctypes.c_int),
        ("rec_height", ctypes.c_int),
        ("rec_max_width", ctypes.c_int),
        ("rec_batch_size", ctypes.c_int),
        ("rec_batch_width_bucket", ctypes.c_int),
        ("rec_parallelism", ctypes.c_int),
        ("hybrid_graph_parallelism", ctypes.c_int),
        ("det_batch_size", ctypes.c_int),
        ("batch_preprocess_parallelism", ctypes.c_int),
        ("image_batch_parallelism", ctypes.c_int),
    ]


class Image(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("rgb", ctypes.POINTER(ctypes.c_uint8)),
    ]


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class TextBox(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("confidence", ctypes.c_float),
        ("bbox_x", ctypes.c_int),
        ("bbox_y", ctypes.c_int),
        ("bbox_w", ctypes.c_int),
        ("bbox_h", ctypes.c_int),
        ("box", Point * 4),
    ]


class Result(ctypes.Structure):
    _fields_ = [("items", ctypes.POINTER(TextBox)), ("count", ctypes.c_size_t)]


class CpuInfo(ctypes.Structure):
    _fields_ = [
        ("avx2_compiled", ctypes.c_int),
        ("avx512_compiled", ctypes.c_int),
        ("neon_compiled", ctypes.c_int),
        ("avx2", ctypes.c_int),
        ("avx512", ctypes.c_int),
        ("neon", ctypes.c_int),
        ("threads", ctypes.c_int),
        ("active_isa", ctypes.c_char * 16),
    ]


def _candidate_libraries():
    env = os.environ.get("PPOCR_LIBRARY")
    if env:
        yield Path(env)
    bundled = Path(__file__).resolve().parent
    names = ("libppocr.so", "libppocr.dylib", "ppocr.dll")
    for name in names:
        yield bundled / name
    yield from bundled.glob("libppocr.so.*")
    found = ctypes.util.find_library("ppocr")
    if found:
        yield Path(found)


def load_library():
    errors = []
    for path in _candidate_libraries():
        if not path or not Path(path).exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        _bind(lib)
        return lib
    hint = "; ".join(errors) if errors else "library not found"
    raise ImportError(
        "Could not load libppocr. Build it with CMake, set PPOCR_LIBRARY, "
        f"or install the wheel. ({hint})"
    )


def _bind(lib: ctypes.CDLL) -> None:
    lib.ppocr_options_init.argtypes = [ctypes.POINTER(Options)]
    lib.ppocr_options_init.restype = None
    lib.ppocr_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(Options),
    ]
    lib.ppocr_create.restype = ctypes.c_void_p
    lib.ppocr_destroy.argtypes = [ctypes.c_void_p]
    lib.ppocr_destroy.restype = None
    lib.ppocr_set_det_thresholds.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.ppocr_set_det_thresholds.restype = ctypes.c_int
    lib.ppocr_recognize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Image),
        ctypes.POINTER(Result),
    ]
    lib.ppocr_recognize.restype = ctypes.c_int
    lib.ppocr_result_free.argtypes = [ctypes.POINTER(Result)]
    lib.ppocr_result_free.restype = None
    lib.ppocr_last_error.argtypes = []
    lib.ppocr_last_error.restype = ctypes.c_char_p
    lib.ppocr_query_cpu_info.argtypes = [ctypes.POINTER(CpuInfo)]
    lib.ppocr_query_cpu_info.restype = None


class NativeOCR:
    def __init__(self, det: str, rec: str, dictionary: str, options: Options):
        self.lib = load_library()
        handle = self.lib.ppocr_create(
            det.encode(), rec.encode(), dictionary.encode(), ctypes.byref(options)
        )
        if not handle:
            raise RuntimeError(self._error() or "ppocr_create failed")
        self.handle = handle

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.ppocr_destroy(self.handle)
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def set_det_thresholds(self, det_threshold: float, box_threshold: float, unclip_ratio: float) -> None:
        status = self.lib.ppocr_set_det_thresholds(
            self.handle, float(det_threshold), float(box_threshold), float(unclip_ratio)
        )
        if status != PPOCR_OK:
            raise RuntimeError(self._error() or "ppocr_set_det_thresholds failed")

    def recognize(self, rgb: np.ndarray):
        array = np.ascontiguousarray(rgb, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("native OCR expects an HWC RGB uint8 image")
        height, width = array.shape[:2]
        image = Image(width, height, array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        result = Result()
        status = self.lib.ppocr_recognize(self.handle, ctypes.byref(image), ctypes.byref(result))
        try:
            if status != PPOCR_OK:
                raise RuntimeError(self._error() or "ppocr_recognize failed")
            boxes = []
            txts = []
            scores = []
            for index in range(int(result.count)):
                item = result.items[index]
                txts.append(item.text.decode("utf-8", errors="replace") if item.text else "")
                scores.append(float(item.confidence))
                boxes.append([[item.box[k].x, item.box[k].y] for k in range(4)])
            return boxes, txts, scores
        finally:
            self.lib.ppocr_result_free(ctypes.byref(result))

    def _error(self) -> str:
        raw = self.lib.ppocr_last_error()
        return raw.decode() if raw else ""


def cpu_info() -> dict:
    lib = load_library()
    info = CpuInfo()
    lib.ppocr_query_cpu_info(ctypes.byref(info))
    return {
        "active_isa": info.active_isa.decode(),
        "avx2": bool(info.avx2),
        "avx512": bool(info.avx512),
        "neon": bool(info.neon),
        "threads": int(info.threads),
        "avx2_compiled": bool(info.avx2_compiled),
        "avx512_compiled": bool(info.avx512_compiled),
        "neon_compiled": bool(info.neon_compiled),
    }


def default_options() -> Options:
    lib = load_library()
    options = Options()
    lib.ppocr_options_init(ctypes.byref(options))
    return options
