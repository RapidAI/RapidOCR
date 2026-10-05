"""RapidOCR 5 (``rapidocr`` 5.0.0a1).

This is the next major version of the ``rapidocr`` package. The import name
is still ``rapidocr``. The 3.x line lives on ``main``; see ``python/README.md``
for what changed.
"""

__version__ = "5.0.0a1"

from .download import download_models
from .main import RapidOCR, RapidOCRError
from .typings import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion
from .utils.load_image import LoadImageError
from .utils.vis_res import VisRes
from ._native import cpu_info

__all__ = [
    "__version__",
    "RapidOCR",
    "RapidOCRError",
    "LoadImageError",
    "VisRes",
    "EngineType",
    "LangCls",
    "LangDet",
    "LangRec",
    "ModelType",
    "OCRVersion",
    "download_models",
    "cpu_info",
]
