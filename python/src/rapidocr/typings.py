"""Enums carried forward from rapidocr 3.x on ``main``.

This module is part of rapidocr 5. ``EngineType.ONNXRUNTIME`` selects this
line's built-in ONNX interpreter.
The other engine names are accepted by the type system so existing imports
keep working; constructing an engine with them raises ``NotImplementedError``.
"""

from enum import Enum


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"
    XPU = "xpu"
    MLU = "mlu"
    DCU = "dcu"
    GCU = "gcu"
    MPS = "mps"


class LangDet(Enum):
    CH = "ch"
    EN = "en"
    MULTI = "multi"


class LangCls(Enum):
    CH = "ch"


class LangRec(Enum):
    CH = "ch"
    CH_DOC = "ch_doc"
    EN = "en"
    ARABIC = "arabic"
    CHINESE_CHT = "chinese_cht"
    CYRILLIC = "cyrillic"
    DEVANAGARI = "devanagari"
    JAPAN = "japan"
    KOREAN = "korean"
    KA = "ka"
    LATIN = "latin"
    TA = "ta"
    TE = "te"
    ESLAV = "eslav"
    TH = "th"
    EL = "el"


class OCRVersion(Enum):
    PPOCRV4 = "PP-OCRv4"
    PPOCRV5 = "PP-OCRv5"
    PPOCRV6 = "PP-OCRv6"


class EngineType(Enum):
    ONNXRUNTIME = "onnxruntime"
    PPOCR_CPP = "ppocr_cpp"
    OPENVINO = "openvino"
    PADDLE = "paddle"
    TORCH = "torch"
    TENSORRT = "tensorrt"
    MNN = "mnn"


class ModelType(Enum):
    MOBILE = "mobile"
    SERVER = "server"
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"


class TaskType(Enum):
    DET = "det"
    CLS = "cls"
    REC = "rec"
