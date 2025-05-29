# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from enum import Enum


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


class OCRVersion(Enum):
    PPOCRV4 = "PP-OCRv4"
    PPOCRV5 = "PP-OCRv5"


class EngineType(Enum):
    ONNXRUNTIME = "onnxruntime"
    OPENVINO = "openvino"
    PADDLE = "paddle"
    TORCH = "torch"


class ModelType(Enum):
    MOBILE = "mobile"
    SERVER = "server"


class TaskType(Enum):
    DET = "det"
    CLS = "cls"
    REC = "rec"
