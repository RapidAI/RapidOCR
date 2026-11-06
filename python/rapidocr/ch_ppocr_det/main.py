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
from typing import Any, Dict

import cv2
import numpy as np

from rapidocr.inference_engine.base import get_engine

from .utils import DBPostProcess, DetPreProcess, TextDetOutput
from ..utils.tiling import merge_tiled_boxes, slice_image

_BOX_SORT_Y_THRESHOLD = 10


class TextDetector:
    # Defaults live at class level as well: some callers build the detector with
    # `__new__` and never run `__init__` (the preprocessing tests do), and the
    # tiling gate must not raise AttributeError on those instances.
    use_tiling: bool = False
    tile_size: int = 1280
    tile_overlap: float = 0.15
    tile_scale: float = 1.0
    tile_min_side: int = 2000

    def __init__(self, cfg: Dict[str, Any]):
        self.limit_side_len = cfg.get("limit_side_len")
        self.limit_type = cfg.get("limit_type")
        self.mean = cfg.get("mean")
        self.std = cfg.get("std")
        self.use_tiling = cfg.get("use_tiling", True)
        self.tile_size = cfg.get("tile_size", 1280)
        self.tile_overlap = cfg.get("tile_overlap", 0.15)
        self.tile_scale = cfg.get("tile_scale", 1.0)
        self.tile_min_side = cfg.get("tile_min_side", 2000)
        self.preprocess_op = None

        post_process = {
            "thresh": cfg.get("thresh", 0.3),
            "box_thresh": cfg.get("box_thresh", 0.5),
            "max_candidates": cfg.get("max_candidates", 1000),
            "unclip_ratio": cfg.get("unclip_ratio", 1.6),
            "use_dilation": cfg.get("use_dilation", True),
            "score_mode": cfg.get("score_mode", "fast"),
        }
        self.postprocess_op = DBPostProcess(**post_process)

        self.session = get_engine(cfg.engine_type)(cfg)

    def __call__(self, img: np.ndarray) -> TextDetOutput:
        start_time = time.perf_counter()

        if img is None:
            raise ValueError("img is None")

        ori_img_shape = img.shape[0], img.shape[1]
        h, w = ori_img_shape

        # Sliced inference (SAHI) for large images (blueprints, big tables).
        # The whole page at native resolution is expensive to push through the
        # network and its small text survives better when the detector only has
        # to look at tile-sized windows.
        #
        # Only for genuinely large pages: a text line crossing a tile boundary
        # is detected as two fragments, which costs more than the extra recall
        # gains on ordinary documents.
        if self.use_tiling and max(h, w) >= self.tile_min_side:
            tiles = slice_image(img, tile_size=self.tile_size, overlap=self.tile_overlap)
            tiled_results = []
            for tile, off_x, off_y in tiles:
                # Small text (CAD dimensions, dense notes) is often below what
                # the detector resolves at native scale; upscaling the tile
                # before inference recovers it. Boxes are mapped back after.
                if self.tile_scale != 1.0:
                    tile = cv2.resize(
                        tile,
                        (int(tile.shape[1] * self.tile_scale), int(tile.shape[0] * self.tile_scale)),
                        interpolation=cv2.INTER_CUBIC,
                    )
                preprocess_op = self.get_preprocess()
                prepro_img = preprocess_op(tile)
                if prepro_img is None:
                    tiled_results.append(([], off_x, off_y))
                    continue
                preds = self.session(prepro_img)
                boxes, scores = self.postprocess_op(preds, tile.shape[:2])
                if boxes is None or len(boxes) < 1:
                    tiled_results.append(([], off_x, off_y))
                    continue
                scale_back = 1.0 / self.tile_scale
                tile_boxes = []
                for box, score in zip(boxes, scores):
                    # Keep the four corners: a rotated quadrilateral cannot be
                    # rebuilt from its axis-aligned envelope, and slanted text is
                    # common on the pages this path exists for.
                    polygon = np.asarray(box, dtype=np.float32) * scale_back
                    xs, ys = polygon[:, 0], polygon[:, 1]
                    tile_boxes.append(
                        (
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max()),
                            float(ys.max()),
                            float(score),
                            polygon,
                        )
                    )
                tiled_results.append((tile_boxes, off_x, off_y))

            merged = merge_tiled_boxes(tiled_results, iou_threshold=0.5)
            if not merged:
                return TextDetOutput()

            boxes = np.array(
                [
                    b[5]
                    if len(b) > 5 and b[5] is not None
                    else [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
                    for b in merged
                ],
                dtype=np.float32,
            )
            order = self.sort_order(boxes)
            boxes = boxes[order]
            scores = [merged[i][4] for i in order]
            elapse = time.perf_counter() - start_time
            return TextDetOutput(img, boxes, scores, elapse=elapse)

        self.preprocess_op = self.get_preprocess()
        prepro_img = self.preprocess_op(img)
        if prepro_img is None:
            return TextDetOutput()

        preds = self.session(prepro_img)
        boxes, scores = self.postprocess_op(preds, ori_img_shape)
        if len(boxes) < 1:
            return TextDetOutput()

        order = self.sort_order(boxes)
        boxes = boxes[order]
        scores = [scores[i] for i in order]
        elapse = time.perf_counter() - start_time
        return TextDetOutput(img, boxes, scores, elapse=elapse)

    def get_preprocess(self) -> DetPreProcess:
        return DetPreProcess(self.limit_side_len, self.limit_type, self.mean, self.std)

    @staticmethod
    def sort_order(dt_boxes: np.ndarray) -> np.ndarray:
        """Permutation that sorts boxes into reading order.

        Exposed separately so the confidence scores can be reordered with the
        boxes: applying the sort to the boxes alone leaves each score attached to
        a different detection.
        """
        if len(dt_boxes) == 0:
            return np.empty(0, dtype=np.int64)

        # Step 1: Stable sort by y (top to bottom)
        y_coords = dt_boxes[:, 0, 1]
        y_order = np.argsort(y_coords, kind="stable")
        boxes_y_sorted = dt_boxes[y_order]
        y_sorted = y_coords[y_order]

        # Step 2: Assign line IDs based on adjacent y differences
        dy = np.diff(y_sorted)
        line_increments = (dy >= _BOX_SORT_Y_THRESHOLD).astype(np.int32)
        line_ids = np.concatenate([[0], np.cumsum(line_increments)])

        # Now, within each line_id group, sort by x (left to right)
        x_coords = boxes_y_sorted[:, 0, 0]
        final_order_in_y_sorted = np.lexsort((x_coords, line_ids))

        return y_order[final_order_in_y_sorted]

    @staticmethod
    def sorted_boxes(dt_boxes: np.ndarray) -> np.ndarray:
        """Equivalent NumPy implementation of the original bubble-adjusted sort."""
        if len(dt_boxes) == 0:
            return dt_boxes
        return dt_boxes[TextDetector.sort_order(dt_boxes)]
