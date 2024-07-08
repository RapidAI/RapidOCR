# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import base64
import copy
import json
from collections import namedtuple
from functools import reduce
from typing import List, Tuple, Union

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


class OCRWebUtils:
    def __init__(self) -> None:
        self.ocr = RapidOCR()
        self.WebReturn = namedtuple(
            "WebReturn",
            ["image", "total_elapse", "elapse_part", "rec_res", "det_boxes"],
        )

    def __call__(self, img_content: str) -> namedtuple:
        if img_content is None:
            raise ValueError("img is None")
        img = self.prepare_img(img_content)
        ocr_res, elapse = self.ocr(img)
        return self.get_web_result(img, ocr_res, elapse)

    def prepare_img(self, img_str: str) -> np.ndarray:
        img_str = img_str.split(",")[1]
        image = base64.b64decode(img_str + "=" * (-len(img_str) % 4))
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def get_web_result(
        self, img: np.ndarray, ocr_res: List, elapse: List
    ) -> Tuple[Union[str, List, str, str]]:
        if ocr_res is None:
            total_elapse, elapse_part = 0, ""
            img_str = self.img_to_base64(img)
            rec_res = json.dumps([], indent=2, ensure_ascii=False)
            boxes = ""
        else:
            boxes, txts, scores = list(zip(*ocr_res))
            scores = [f"{v:.4f}" for v in scores]
            rec_res = list(zip(range(len(txts)), txts, scores))
            rec_res = json.dumps(rec_res, indent=2, ensure_ascii=False)

            det_im = self.draw_text_det_res(np.array(boxes), img)
            img_str = self.img_to_base64(det_im)

            total_elapse = reduce(lambda x, y: float(x) + float(y), elapse)
            elapse_part = ",".join([f"{x:.4f}" for x in elapse])

        web_return = self.WebReturn(
            image=img_str,
            total_elapse=f"{total_elapse:.4f}",
            elapse_part=elapse_part,
            rec_res=rec_res,
            det_boxes=boxes,
        )
        return json.dumps(web_return._asdict())

    @staticmethod
    def img_to_base64(img) -> str:
        img = cv2.imencode(".png", img)[1]
        img_str = str(base64.b64encode(img))[2:-1]
        return img_str

    @staticmethod
    def draw_text_det_res(dt_boxes: np.ndarray, raw_im: np.ndarray) -> np.ndarray:
        src_im = copy.deepcopy(raw_im)
        for i, box in enumerate(dt_boxes):
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(src_im, [box], True, color=(0, 0, 255), thickness=1)
            cv2.putText(
                src_im,
                str(i),
                (int(box[0][0]), int(box[0][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
        return src_im
