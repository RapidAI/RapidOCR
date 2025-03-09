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
import copy
import math
import time
from typing import Any, Dict, List, Union

import cv2
import numpy as np

from rapidocr.inference_engine.base import get_engine

from .utils import ClsPostProcess, TextClsOutput


class TextClassifier:
    def __init__(self, config: Dict[str, Any]):
        self.cls_image_shape = config["cls_image_shape"]
        self.cls_batch_num = config["cls_batch_num"]
        self.cls_thresh = config["cls_thresh"]
        self.postprocess_op = ClsPostProcess(config["label_list"])

        self.session = get_engine(config.engine_name)(config)

    def __call__(self, img_list: Union[np.ndarray, List[np.ndarray]]) -> TextClsOutput:
        if isinstance(img_list, np.ndarray):
            img_list = [img_list]

        img_list = copy.deepcopy(img_list)

        # Calculate the aspect ratio of all text bars
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        img_num = len(img_list)
        cls_res = [("", 0.0)] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)

            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)

            starttime = time.time()
            prob_out = self.session(norm_img_batch)
            cls_result = self.postprocess_op(prob_out)
            elapse += time.time() - starttime

            for rno, (label, score) in enumerate(cls_result):
                cls_res[indices[beg_img_no + rno]] = (label, score)
                if "180" in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1
                    )
        return TextClsOutput(img_list=img_list, cls_res=cls_res, elapse=elapse)

    def resize_norm_img(self, img: np.ndarray) -> np.ndarray:
        img_c, img_h, img_w = self.cls_image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(img_h * ratio) > img_w:
            resized_w = img_w
        else:
            resized_w = int(math.ceil(img_h * ratio))

        resized_image = cv2.resize(img, (resized_w, img_h))
        resized_image = resized_image.astype("float32")
        if img_c == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255

        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        return padding_im
