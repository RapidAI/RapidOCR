# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from ..utils.logger import Logger
from ..utils.utils import save_img
from ..utils.vis_res import VisRes

logger = Logger(logger_name=__name__).get_log()


@dataclass
class TextClsOutput:
    img_list: Optional[List[np.ndarray]] = None
    cls_res: Optional[List[Tuple[str, float]]] = None
    elapse: Optional[float] = None

    def __len__(self):
        if self.img_list is None:
            return 0
        return len(self.img_list)

    def vis(self, save_path: Optional[Union[str, Path]] = None) -> Optional[np.ndarray]:
        if self.img_list is None or self.cls_res is None:
            logger.warning("No image or txts to visualize.")
            return None

        txts = [f"{txt} {score:.2f}" for txt, score in self.cls_res]
        scores = [score for _, score in self.cls_res]

        vis = VisRes()
        vis_img = vis.draw_rec_res(self.img_list, txts, scores)

        if save_path is not None:
            save_img(save_path, vis_img)
            logger.info("Visualization saved as %s", save_path)
        return vis_img


class ClsPostProcess:
    def __init__(self, label_list: List[str]):
        self.label_list = label_list

    def __call__(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        pred_idxs = preds.argmax(axis=1)
        decode_out = [
            (self.label_list[int(idx)], preds[i, int(idx)])
            for i, idx in enumerate(pred_idxs)
        ]
        return decode_out
