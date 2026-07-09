from typing import List

from .main import RapidOCR as RapidOCR
from .utils.download_models import download_models as download_models
from .utils.load_image import LoadImageError as LoadImageError
from .utils.typings import EngineType as EngineType
from .utils.typings import LangCls as LangCls
from .utils.typings import LangDet as LangDet
from .utils.typings import LangRec as LangRec
from .utils.typings import ModelType as ModelType
from .utils.typings import OCRVersion as OCRVersion
from .utils.vis_res import VisRes as VisRes

__all__: List[str]
