# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .logger import Logger
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

    def vis(self):
        if self.img is None or self.boxes is None:
            logger.warning("No image or boxes to visualize.")
            return

        vis = VisRes()
        if all(v is None for v in self.word_results):
            vis_img = vis(self.img, self.boxes, self.txts, self.scores)
            cv2.imwrite("vis.png", vis_img)
            logger.info("Visualization saved as vis.png.")
            return

        # single word vis
        words_results = self.word_results
        words, words_scores, words_boxes = list(zip(*words_results))
        vis_img = vis(self.img, words_boxes, words, words_scores)
        cv2.imwrite("vis_single.png", vis_img)
        logger.info("Single word visualization saved as vis_single.png.")
        return
