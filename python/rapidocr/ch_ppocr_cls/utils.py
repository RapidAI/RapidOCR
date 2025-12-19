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

import cv2
import numpy as np

from ..utils.log import logger
from ..utils.utils import save_img
from ..utils.vis_res import VisRes


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

        vis = VisRes()

        txts = [f"{txt} {score:.2f}" for txt, score in self.cls_res]
        img_degrees, scores = list(zip(*self.cls_res))

        raw_img_list = self.restore_image_orientation(self.img_list, img_degrees)
        vis_img = vis.draw_rec_res(raw_img_list, txts, list(scores))

        if save_path is not None:
            save_img(save_path, vis_img)
            logger.info("Visualization saved as %s", save_path)
        return vis_img

    def restore_image_orientation(
        self, img_list: List[np.ndarray], img_degrees: Tuple[str]
    ):
        results = []
        for img, rotate_degree in zip(img_list, img_degrees):
            if rotate_degree != "180":
                results.append(img)
                continue

            rotate_img = cv2.rotate(img, 1)
            results.append(rotate_img)
        return results


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
