# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import copy

import cv2
import numpy as np

from functools import lru_cache
from utils.config import conf
from utils.utils import Ticker
from .classify import TextClassifier
from .detect import TextDetector
from .recognize import TextRecognizer


def get_rotate_crop_image(img, points):
    """根据box定义, 从图像中截取相应的部分, 通过透视变换转换为标准长方形图像"""
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    transform = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        transform,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[:2]
    # 将竖向的文字方向转为横向, 仅当 高>1.5*宽 时进行转换
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


@lru_cache(maxsize=None)
def load_onnx_model(step, name):
    model_config = conf["models"][step][name]
    model_class = {
        "detect": TextDetector,
        "classify": TextClassifier,
        "recognize": TextRecognizer,
    }[step]
    return model_class(model_config["path"], model_config.get("config"))


class RapidOCR:
    def __init__(self, config):
        super(RapidOCR).__init__()
        self.config = config
        self.text_score = config["config"]["text_score"]
        self.min_height = config["config"]["min_height"]

        models = config["models"]
        self.text_detector = load_onnx_model("detect", models["detect"])
        self.text_recognizer = load_onnx_model("recognize", models["recognize"])
        self.text_cls = load_onnx_model("classify", models["classify"])

    def __call__(self, img: np.ndarray, detect=True, classify=True):
        ticker = Ticker()
        h, w = img.shape[:2]
        if not detect or h < self.min_height:
            dt_boxes, img_crop_list = self.get_boxes_img_without_det(img, h, w)
            ticker.tick("detect")
        else:
            dt_boxes = self.text_detector(img)
            ticker.tick("detect")

            if dt_boxes is None or len(dt_boxes) < 1:
                return [], ticker.maps
            if conf["global"]["verbose"]:
                print(f"boxes num: {len(dt_boxes)}")

            dt_boxes = self.sorted_boxes(dt_boxes)

            img_crop_list = self.get_crop_img_list(img, dt_boxes)
            ticker.tick("post-detect")

        if classify:
            # 进行子图像角度修正
            img_crop_list, _ = self.text_cls(img_crop_list)
            ticker.tick("classify")
            if conf["global"]["verbose"]:
                print(f"cls num: {len(img_crop_list)}")

        recog_result = self.text_recognizer(img_crop_list)
        ticker.tick("recognize")
        if conf["global"]["verbose"]:
            print(f"rec_res num: {len(recog_result)}")

        results = self.filter_boxes_rec_by_score(dt_boxes, recog_result)
        ticker.tick("post-recognize")
        return results, ticker.maps

    def get_boxes_img_without_det(self, img, h, w):
        x0, y0, x1, y1 = 0, 0, w, h
        dt_boxes = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        dt_boxes = dt_boxes[np.newaxis, ...]
        img_crop_list = [img]
        return dt_boxes, img_crop_list

    def get_crop_img_list(self, img, dt_boxes):
        img_crop_list = []
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        return img_crop_list

    @staticmethod
    def sorted_boxes(dt_boxes):
        """对文本框检测结果进行排序, 调整为从上到下、从左到右

        args:
            dt_boxes(array): detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """

        class AlignBox:
            def __init__(self, data) -> None:
                self.data = data
                self.x = data[0][0]
                self.y = data[0][1]

            def __lt__(self, other: "AlignBox"):
                dy = self.y - other.y
                # y差距小于10, 视为相等, 根据x排序
                if abs(dy) < 10:
                    return self.x < other.x
                # 否则根据y排序
                return dy < 0

        align_boxes = sorted([AlignBox(b) for b in dt_boxes])
        return [b.data for b in align_boxes]

    def filter_boxes_rec_by_score(self, dt_boxes, rec_res):
        results = []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.text_score:
                results.append({"box": box, "text": text, "score": score})
        return results
