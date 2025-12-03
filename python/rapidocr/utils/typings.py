# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from enum import Enum


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"


class LangDet(str, Enum):
    CH = "ch"
    EN = "en"
    MULTI = "multi"


class LangCls(str, Enum):
    CH = "ch"


class LangRec(str, Enum):
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


class OCRVersion(str, Enum):
    PPOCRV4 = "PP-OCRv4"
    PPOCRV5 = "PP-OCRv5"


class EngineType(str, Enum):
    ONNXRUNTIME = "onnxruntime"
    OPENVINO = "openvino"
    PADDLE = "paddle"
    TORCH = "torch"


class ModelType(str, Enum):
    MOBILE = "mobile"
    SERVER = "server"


class TaskType(str, Enum):
    DET = "det"
    CLS = "cls"
    REC = "rec"
