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
import argparse
import math
import time
from typing import List
from pathlib import Path

import cv2
import numpy as np

try:
    from .utils import CTCLabelDecode, OpenVINOInferSession
except:
    from utils import CTCLabelDecode, OpenVINOInferSession


class TextRecognizer():
    def __init__(self, config):
        self.rec_image_shape = config['rec_img_shape']
        self.rec_batch_num = config['rec_batch_num']

        dict_path = str(Path(__file__).parent / 'ppocr_keys_v1.txt')
        self.character_dict_path =  config.get( 'keys_path', dict_path)
        self.postprocess_op = CTCLabelDecode(self.character_dict_path)

        openvino_instance = OpenVINOInferSession(config)
        self.session = openvino_instance.session

    def __call__(self, img_list: List[np.ndarray]):
        if isinstance(img_list, np.ndarray):
            img_list = [img_list]

        # Calculate the aspect ratio of all text bars
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        img_num = len(img_list)
        rec_res = [['', 0.0]] * img_num

        batch_num = self.rec_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)

            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img_batch.append(norm_img[np.newaxis, :])
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)

            starttime = time.time()
            self.session.infer(inputs=[norm_img_batch])
            preds = self.session.get_output_tensor().data

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            elapse += time.time() - starttime
        return rec_res, elapse

    def resize_norm_img(self, img, max_wh_ratio):
        img_channel, img_height, img_width = self.rec_image_shape
        assert img_channel == img.shape[2]

        img_width = int(img_height * max_wh_ratio)

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(img_height * ratio) > img_width:
            resized_w = img_width
        else:
            resized_w = int(math.ceil(img_height * ratio))

        resized_image = cv2.resize(img, (resized_w, img_height))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((img_channel, img_height, img_width),
                              dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='image_dir|image_path')
    parser.add_argument('--config_path', type=str, default='config.yaml')
    args = parser.parse_args()

    from utils import read_yaml
    config = read_yaml(args.config_path)
    text_recognizer = TextRecognizer(config)

    img = cv2.imread(args.image_path)
    rec_res, predict_time = text_recognizer(img)
    print(f'rec result: {rec_res}\t cost: {predict_time}s')
