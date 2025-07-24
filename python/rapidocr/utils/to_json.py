# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from typing import Any, Dict, List, Tuple

import numpy as np


class ToJSON:
    @classmethod
    def to(
        cls, boxes: np.ndarray, txts: Tuple[str], scores: Tuple[float]
    ) -> List[Dict[Any, Any]]:
        results = []
        for box, txt, score in zip(boxes, txts, scores):
            results.append({"box": box.tolist(), "txt": txt, "score": score})
        return results
