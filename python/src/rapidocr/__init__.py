from .download import download_models
from .main import LoadImageError, RapidOCR, RapidOCRError
from .typings import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion
from ._native import cpu_info

__all__ = [
    "RapidOCR",
    "RapidOCRError",
    "LoadImageError",
    "EngineType",
    "LangCls",
    "LangDet",
    "LangRec",
    "ModelType",
    "OCRVersion",
    "download_models",
    "cpu_info",
]
