# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .logger import Logger
from .to_json import ToJSON
from .to_markdown import ToMarkdown
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
    viser: Optional[VisRes] = None

    def __post_init__(self):
        self.elapse = sum(v for v in self.elapse_list if isinstance(v, float))

    def __len__(self):
        if self.txts is None:
            return 0
        return len(self.txts)

    def to_json(self) -> Optional[List[Dict[Any, Any]]]:
        if any(v is None for v in (self.boxes, self.txts, self.scores)):
            logger.warning("The identified content is empty.")
            return None
        return ToJSON.to(self.boxes, self.txts, self.scores)

    def to_markdown(self) -> str:
        return ToMarkdown.to(self.boxes, self.txts)

    def vis(self, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        if self.img is None or self.boxes is None:
            logger.warning("No image or boxes to visualize.")
            return None

        if self.viser is None:
            logger.error("vis instance is None")
            return None

        if all(v is None for v in self.word_results):
            vis_img = self.viser(self.img, self.boxes, self.txts, self.scores)

            if save_path is not None:
                save_img(save_path, vis_img)
                logger.info("Visualization saved as %s", save_path)
            return vis_img

        # single word vis
        words_results = sum(self.word_results, ())
        words, words_scores, words_boxes = list(zip(*words_results))
        vis_img = self.viser(self.img, words_boxes, words, words_scores)

        if save_path is not None:
            save_img(save_path, vis_img)
            logger.info("Single word visualization saved as %s", save_path)
        return vis_img
