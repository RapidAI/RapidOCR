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
# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import time
from typing import Any, Dict, List

import numpy as np

from rapidocr.inference_engine.base import get_engine

from .utils import DBPostProcess, DetPreProcess, TextDetOutput


class TextDetector:
    def __init__(self, config: Dict[str, Any]):
        self.limit_side_len = config.get("limit_side_len")
        self.limit_type = config.get("limit_type")
        self.mean = config.get("mean")
        self.std = config.get("std")
        self.preprocess_op = None

        post_process = {
            "thresh": config.get("thresh", 0.3),
            "box_thresh": config.get("box_thresh", 0.5),
            "max_candidates": config.get("max_candidates", 1000),
            "unclip_ratio": config.get("unclip_ratio", 1.6),
            "use_dilation": config.get("use_dilation", True),
            "score_mode": config.get("score_mode", "fast"),
        }
        self.postprocess_op = DBPostProcess(**post_process)

        self.session = get_engine(config.engine_name)(config)

    def __call__(self, img: np.ndarray) -> TextDetOutput:
        start_time = time.perf_counter()

        if img is None:
            raise ValueError("img is None")

        ori_img_shape = img.shape[0], img.shape[1]
        self.preprocess_op = self.get_preprocess(max(img.shape[0], img.shape[1]))
        prepro_img = self.preprocess_op(img)
        if prepro_img is None:
            return TextDetOutput()

        preds = self.session(prepro_img)
        boxes, scores = self.postprocess_op(preds, ori_img_shape)
        if len(boxes) < 1:
            return TextDetOutput()

        boxes = self.sorted_boxes(boxes)
        elapse = time.perf_counter() - start_time
        return TextDetOutput(boxes, scores, elapse=elapse)

    def get_preprocess(self, max_wh: int) -> DetPreProcess:
        if self.limit_type == "min":
            limit_side_len = self.limit_side_len
        elif max_wh < 960:
            limit_side_len = 960
        elif max_wh < 1500:
            limit_side_len = 1500
        else:
            limit_side_len = 2000
        return DetPreProcess(limit_side_len, self.limit_type, self.mean, self.std)

    @staticmethod
    def sorted_boxes(dt_boxes: np.ndarray) -> List[np.ndarray]:
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if (
                    abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10
                    and _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes
