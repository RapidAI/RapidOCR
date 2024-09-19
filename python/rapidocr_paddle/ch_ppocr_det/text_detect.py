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
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rapidocr_paddle.utils import PaddleInferSession

from .utils import DBPostProcess, DetPreProcess


class TextDetector:
    def __init__(self, config: Dict[str, Any]):
        limit_side_len = config.get("limit_side_len", 736)
        limit_type = config.get("limit_type", "min")
        self.preprocess_op = DetPreProcess(limit_side_len, limit_type)

        post_process = {
            "thresh": config.get("thresh", 0.3),
            "box_thresh": config.get("box_thresh", 0.5),
            "max_candidates": config.get("max_candidates", 1000),
            "unclip_ratio": config.get("unclip_ratio", 1.6),
            "use_dilation": config.get("use_dilation", True),
            "score_mode": config.get("score_mode", "fast"),
        }
        self.postprocess_op = DBPostProcess(**post_process)

        self.infer = PaddleInferSession(config)

    def __call__(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        start_time = time.perf_counter()

        if img is None:
            raise ValueError("img is None")

        ori_img_shape = img.shape[0], img.shape[1]
        prepro_img = self.preprocess_op(img)
        if prepro_img is None:
            return None, 0

        preds = self.infer(prepro_img)[0]
        dt_boxes, dt_boxes_scores = self.postprocess_op(preds, ori_img_shape)
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_img_shape)
        elapse = time.perf_counter() - start_time
        return dt_boxes, elapse

    def filter_tag_det_res(
        self, dt_boxes: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        img_height, img_width = image_shape
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)

            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue

            dt_boxes_new.append(box)
        return np.array(dt_boxes_new)

    def order_points_clockwise(self, pts: np.ndarray) -> np.ndarray:
        """
        reference from:
        https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(
        self, points: np.ndarray, img_height: int, img_width: int
    ) -> np.ndarray:
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points
