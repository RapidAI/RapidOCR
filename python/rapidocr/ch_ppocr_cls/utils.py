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
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TextClsOutput:
    img_list: Optional[List[np.ndarray]] = None
    cls_res: Optional[List[Tuple[str, float]]] = None
    elapse: Optional[float] = None

    def __len__(self):
        if self.img_list is None:
            return 0
        return len(self.img_list)


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
