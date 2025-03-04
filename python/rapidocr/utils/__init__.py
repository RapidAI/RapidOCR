# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from .load_image import LoadImage, LoadImageError
from .logger import Logger
from .parse_parameters import ParseLang, ParseParams
from .process_img import add_round_letterbox, increase_min_side, reduce_max_side
from .typings import RapidOCROutput
from .utils import download_file
from .vis_res import VisRes
