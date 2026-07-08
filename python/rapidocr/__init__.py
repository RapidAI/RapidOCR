# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from .utils.typings import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion


def __getattr__(name):
    if name == "RapidOCR":
        from .main import RapidOCR

        return RapidOCR

    if name == "LoadImageError":
        from .utils.load_image import LoadImageError

        return LoadImageError

    if name == "VisRes":
        from .utils.vis_res import VisRes

        return VisRes

    if name == "download_models":
        from .utils.download_models import download_models

        return download_models

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RapidOCR",
    "LoadImageError",
    "EngineType",
    "LangCls",
    "LangDet",
    "LangRec",
    "ModelType",
    "OCRVersion",
    "VisRes",
    "download_models",
]
