# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np

from .logger import Logger
from .utils import save_img
from .vis_res import VisRes

logger = Logger(logger_name=__name__).get_log()


@dataclass
class RapidOCROutput:
    img: Optional[np.ndarray] = None
    boxes: Optional[np.ndarray] = None
    txts: Optional[Tuple[str]] = None
    scores: Optional[Tuple[float]] = None
    word_results: Tuple[Tuple[str, float, Optional[List[List[int]]]]] = (
        ("", 1.0, None),
    )
    elapse_list: List[Union[float, None]] = field(default_factory=list)
    elapse: float = field(init=False)
    lang_rec: Optional[str] = None

    def __post_init__(self):
        self.elapse = sum(v for v in self.elapse_list if isinstance(v, float))

    def __len__(self):
        if self.txts is None:
            return 0
        return len(self.txts)

    def to_json(self):
        pass

    def to_paddleocr_format(self):
        """Refer: https://pypi.org/project/paddleocr/2.7.3/
        Return format like:
        [
          [[[6.0, 2.0], [322.0, 9.0], [320.0, 104.0], [4.0, 97.0]], ['正品促销', 0.99893]],
          [[[70.0, 98.0], [252.0, 98.0], [252.0, 125.0], [70.0, 125.0]], ['大桶装更划算', 0.9843]]
        ]
        """
        rec_res = list(zip(self.txts, self.scores))
        dt_boxes = [v.tolist() for v in self.boxes]

        final_res = []
        for box, rec in zip(dt_boxes, rec_res):
            final_res.append([box, list(rec)])
        return final_res

    def vis(self, save_path: Optional[str] = None, font_path: Optional[str] = None):
        if self.img is None or self.boxes is None:
            logger.warning("No image or boxes to visualize.")
            return

        vis = VisRes()
        if all(v is None for v in self.word_results):
            vis_img = vis(
                self.img,
                self.boxes,
                self.txts,
                self.scores,
                font_path=font_path,
                lang_rec=self.lang_rec,
            )

            if save_path is not None:
                save_img(save_path, vis_img)
                logger.info("Visualization saved as %s", save_path)
            return vis_img

        # single word vis
        words_results = self.word_results
        words, words_scores, words_boxes = list(zip(*words_results))
        vis_img = vis(
            self.img,
            words_boxes,
            words,
            words_scores,
            font_path=font_path,
            lang_rec=self.lang_rec,
        )

        if save_path is not None:
            save_img(save_path, vis_img)
            logger.info("Single word visualization saved as %s", save_path)
        return vis_img


class LangDet(Enum):
    CH = "ch"
    EN = "en"
    MULTI = "multi"


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
