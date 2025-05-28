# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from ..utils.logger import Logger
from ..utils.utils import save_img
from ..utils.vis_res import VisRes

logger = Logger(logger_name=__name__).get_log()


@dataclass
class TextRecConfig:
    intra_op_num_threads: int = -1
    inter_op_num_threads: int = -1
    use_cuda: bool = False
    use_dml: bool = False
    model_path: Union[str, Path, None] = None

    rec_batch_num: int = 6
    rec_img_shape: Tuple[int, int, int] = (3, 48, 320)
    rec_keys_path: Union[str, Path, None] = None


@dataclass
class TextRecInput:
    img: Union[np.ndarray, List[np.ndarray], None] = None
    return_word_box: bool = False


@dataclass
class TextRecOutput:
    imgs: Optional[List[np.ndarray]] = None
    txts: Optional[Tuple[str]] = None
    scores: Tuple[float] = (1.0,)
    word_results: Tuple[Tuple[str, float, Optional[List[List[int]]]]] = (
        ("", 1.0, None),
    )
    elapse: Optional[float] = None
    lang_type: Optional[str] = None

    def __len__(self):
        if self.txts is None:
            return 0
        return len(self.txts)

    def vis(self, save_path: Optional[Union[str, Path]] = None) -> Optional[np.ndarray]:
        if self.imgs is None or self.txts is None:
            logger.warning("No image or txts to visualize.")
            return None

        vis = VisRes()
        vis_img = vis.draw_rec_res(
            self.imgs, self.txts, self.scores, lang_type=self.lang_type
        )

        if save_path is not None:
            save_img(save_path, vis_img)
            logger.info("Visualization saved as %s", save_path)
        return vis_img


class WordType(Enum):
    CN = "cn"
    EN = "en"
    NUM = "num"
    EN_NUM = "en&num"


@dataclass
class WordInfo:
    words: List[List[str]] = field(default_factory=list)
    word_cols: List[List[int]] = field(default_factory=list)
    word_types: List[WordType] = field(default_factory=list)
    line_txt_len: float = 0.0
    confs: List[float] = field(default_factory=list)
