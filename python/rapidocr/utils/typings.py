# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np


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
        pass
        # vis = VisRes()
        # if any(v for v in self.word_results if v is not None):
        #     words_results = self.word_results
        #     words, words_scores, words_boxes = list(zip(*words_results))
        #     vis_img = vis(args.img_path, words_boxes, words, words_scores)
        #     save_path = cur_dir / f"{Path(args.img_path).stem}_vis_single.png"
        #     cv2.imwrite(str(save_path), vis_img)
        #     print(f"The vis single result has saved in {save_path}")
