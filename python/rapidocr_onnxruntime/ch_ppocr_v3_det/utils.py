# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
import importlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def create_operators(op_param_dict: Dict[str, Optional[Dict[str, Any]]]) -> List[Any]:
    ops = []
    cur_module = importlib.import_module("rapidocr_onnxruntime.ch_ppocr_v3_det.utils")
    for op_name, param in op_param_dict.items():
        if param is None:
            param = {}

        process_class = getattr(cur_module, op_name)
        op = process_class(**param)
        ops.append(op)
    return ops


def transform(
    data: Dict[str, np.ndarray], ops: Optional[List[Any]] = None
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if ops is None:
        ops = []

    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


class DetResizeForTest:
    def __init__(self, **kwargs):
        self.resize_type = 0
        if "image_shape" in kwargs:
            self.image_shape = kwargs["image_shape"]
            self.resize_type = 1
        elif "limit_side_len" in kwargs:
            self.limit_side_len = kwargs.get("limit_side_len", 736)
            self.limit_type = kwargs.get("limit_type", "min")

        if "resize_long" in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get("resize_long", 960)
        else:
            self.limit_side_len = kwargs.get("limit_side_len", 736)
            self.limit_type = kwargs.get("limit_type", "min")

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        img = data["image"]
        src_h, src_w = img.shape[:2]

        if self.resize_type == 0:
            img = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img = self.resize_image_type2(img)
        else:
            img = self.resize_image_type1(img)

        new_data = {"image": img, "shape": np.array([src_h, src_w])}
        return new_data

    def resize_image_type1(self, img: np.ndarray) -> np.ndarray:
        resize_h, resize_w = self.image_shape
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img

    def resize_image_type0(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        resize image to a size multiple of 32 which is required by the network
        """
        limit_side_len = self.limit_side_len
        h, w = img.shape[:2]

        # limit the max side
        if self.limit_type == "max":
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0
        else:
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.0

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except Exception as exc:
            raise ResizeImgError from exc

        return img

    def resize_image_type2(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        resize_h, resize_w = h, w

        # Fix the longer side
        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img


class NormalizeImage:
    """normalize image such as substract mean, divide std"""

    def __init__(
        self,
        scale: Optional[str] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        order: str = "chw",
    ):
        if isinstance(scale, str):
            scale = eval(scale)

        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        img = np.array(data["image"]).astype(np.float32)
        data["image"] = (img * self.scale - self.mean) / self.std
        return data


class ToCHWImage:
    """convert hwc image to chw image"""

    def __init__(self):
        pass

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        img = np.array(data["image"])
        data["image"] = img.transpose((2, 0, 1))
        return data


class KeepKeys:
    def __init__(self, keep_keys: List[str]):
        self.keep_keys = keep_keys

    def __call__(self, data: Dict[str, np.ndarray]) -> List[np.ndarray]:
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class ResizeImgError(Exception):
    pass


class DBPostProcess:
    """The post process for Differentiable Binarization (DB)."""

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.7,
        max_candidates: int = 1000,
        unclip_ratio: float = 2.0,
        score_mode: str = "fast",
        use_dilation: bool = False,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode

        self.dilation_kernel = None
        if use_dilation:
            self.dilation_kernel = np.array([[1, 1], [1, 1]])

    def __call__(
        self, pred: np.ndarray, shape: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w = shape[batch_index]
            if self.dilation_kernel is None:
                mask = segmentation[batch_index]
            else:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel,
                )

            boxes, scores = self.boxes_from_bitmap(
                pred[batch_index], mask, src_w, src_h
            )

            boxes_batch.append({"points": boxes})
        return boxes_batch

    def boxes_from_bitmap(
        self, pred: np.ndarray, bitmap: np.ndarray, dest_width: int, dest_height: int
    ) -> Tuple[np.ndarray, List[float]]:
        """
        bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        """

        height, width = bitmap.shape

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes, scores = [], []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)

            if self.box_thresh > score:
                continue

            box = self.unclip(points)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.astype(np.int32))
            scores.append(score)
        return np.array(boxes, dtype=np.int32), scores

    def get_mini_boxes(self, contour: np.ndarray) -> Tuple[np.ndarray, float]:
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0

        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = np.array(
            [points[index_1], points[index_2], points[index_3], points[index_4]]
        )
        return box, min(bounding_box[1])

    @staticmethod
    def box_score_fast(bitmap: np.ndarray, _box: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def box_score_slow(self, bitmap: np.ndarray, contour: np.ndarray) -> float:
        """use polyon mean score as the mean score"""
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def unclip(self, box: np.ndarray) -> np.ndarray:
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance)).reshape((-1, 1, 2))
        return expanded
