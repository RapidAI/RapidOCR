# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from importlib import import_module
from typing import TYPE_CHECKING

from .utils.typings import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion

if TYPE_CHECKING:
    from .main import RapidOCR
    from .utils.download_models import download_models
    from .utils.load_image import LoadImageError
    from .utils.vis_res import VisRes

_LAZY_IMPORTS = {
    "RapidOCR": ".main",
    "LoadImageError": ".utils.load_image",
    "VisRes": ".utils.vis_res",
    "download_models": ".utils.download_models",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        return getattr(import_module(_LAZY_IMPORTS[name], __name__), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *__all__])


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
