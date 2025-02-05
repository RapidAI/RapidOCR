# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RapidOCROutput:
    boxes: Optional[List[List[float]]] = None
    txts: Optional[List[str]] = None
    scores: Optional[List[float]] = None
    word_results: Tuple[Tuple[str, float, Optional[List[List[int]]]]] = (
        ("", 1.0, None),
    )
    elapse_list: List[float] = field(default_factory=list)
    elapse: float = field(init=False)

    def __post_init__(self):
        self.elapse = sum(self.elapse_list)

    def to_json(self):
        pass

    def to_paddleocr_format(self):
        """Return format like:
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
