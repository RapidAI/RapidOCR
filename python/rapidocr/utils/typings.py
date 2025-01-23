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
        rec_res = list(zip(self.txts, self.scores))

        print("ok")
