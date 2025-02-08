# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from ..inference_engine.onnxruntime import OrtInferSession
from .load_image import LoadImage, LoadImageError
from .logger import get_logger
from .parse_parameters import UpdateParameters, init_args, parse_lang, update_model_path
from .process_img import add_round_letterbox, increase_min_side, reduce_max_side
from .typings import RapidOCROutput
from .vis_res import VisRes
