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
import time

import cv2
import numpy as np

from rapidocr_onnxruntime.utils import OrtInferSession, read_yaml

from .utils import DBPostProcess, create_operators, transform


class TextDetector():
    def __init__(self, config):
        pre_process_list = {
            'DetResizeForTest': {
                'limit_side_len': config.get('limit_side_len', 736),
                'limit_type': config.get('limit_type', 'min')
            },
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            },
            'ToCHWImage': None,
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }
        self.preprocess_op = create_operators(pre_process_list)

        post_process = {
            'thresh': config.get('thresh', 0.3),
            'box_thresh': config.get('box_thresh', 0.5),
            'max_candidates': config.get('max_candidates', 1000),
            'unclip_ratio': config.get('unclip_ratio', 1.6),
            'use_dilation': config.get('use_dilation', True),
            'score_mode': config.get('score_mode', 'fast'),
        }
        self.postprocess_op = DBPostProcess(**post_process)

        self.infer = OrtInferSession(config)

    def __call__(self, img):
        if img is None:
            raise ValueError('img is None')

        ori_im_shape = img.shape[:2]

        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0

        img = np.expand_dims(img, axis=0).astype(np.float32)
        shape_list = np.expand_dims(shape_list, axis=0)

        starttime = time.time()
        preds = self.infer(img)[0]
        post_result = self.postprocess_op(preds, shape_list)

        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im_shape)
        elapse = time.time() - starttime
        return dt_boxes, elapse

    def order_points_clockwise(self, pts):
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

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml')
    parser.add_argument('--image_path', type=str, default=None)
    args = parser.parse_args()

    config = read_yaml(args.config_path)

    text_detector = TextDetector(config)

    img = cv2.imread(args.image_path)
    dt_boxes, elapse = text_detector(img)

    from utils import draw_text_det_res
    src_im = draw_text_det_res(dt_boxes, args.image_path)
    cv2.imwrite('det_results.jpg', src_im)
    print('The det_results.jpg has been saved in the current directory.')
