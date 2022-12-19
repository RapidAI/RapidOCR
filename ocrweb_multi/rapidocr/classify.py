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
import json
from typing import List

import cv2
import numpy as np
from utils.utils import OrtInferSession


class ClsPostProcess():
    """Convert between text-label and text-index"""

    def __init__(self, label_list):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list

    def __call__(self, preds, label=None):
        pred_idxs = preds.argmax(axis=1)
        decode_out = [
            (self.label_list[idx], preds[i, idx]) for i, idx in enumerate(pred_idxs)
        ]
        if label is None:
            return decode_out

        label = [(self.label_list[idx], 1.0) for idx in label]
        return decode_out, label


class TextClassifier():
    def __init__(self, path, config):
        self.cls_batch_num = config['batch_size']
        self.cls_thresh = config['score_thresh']

        session_instance = OrtInferSession(path)
        self.session = session_instance.session
        metamap = self.session.get_modelmeta().custom_metadata_map

        self.cls_image_shape = json.loads(metamap['shape'])

        labels = json.loads(metamap['labels'])
        self.postprocess_op = ClsPostProcess(labels)
        self.input_name = session_instance.get_input_name()

    def resize_norm_img(self, img):
        img_c, img_h, img_w = self.cls_image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(img_h * ratio) > img_w:
            resized_w = img_w
        else:
            resized_w = int(math.ceil(img_h * ratio))

        resized_image = cv2.resize(img, (resized_w, img_h))
        resized_image = resized_image.astype('float32')
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

    def __call__(self, img_list: List[np.ndarray]):
        if isinstance(img_list, np.ndarray):
            img_list = [img_list]

        img_list = copy.deepcopy(img_list)

        # Calculate the aspect ratio of all text bars
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        img_num = len(img_list)
        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)

            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)

            onnx_inputs = {self.input_name: norm_img_batch}
            prob_out = self.session.run(None, onnx_inputs)[0]
            cls_result = self.postprocess_op(prob_out)

            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if label == '180' and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1
                    )
        return img_list, cls_res
