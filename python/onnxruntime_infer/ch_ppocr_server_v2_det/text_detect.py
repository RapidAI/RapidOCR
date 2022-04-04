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
import argparse
import time

import cv2
import numpy as np
import onnxruntime

try:
    from .utils import (DBPostProcess, check_and_read_gif, create_operators,
                        draw_text_det_res, transform)
except:
    from utils import (DBPostProcess, check_and_read_gif, create_operators,
                       draw_text_det_res, transform)


class TextDetector(object):
    def __init__(self, det_model_path):
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 736,
                'limit_type': 'min'
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = DBPostProcess(thresh=0.3,
                                            box_thresh=0.5,
                                            max_candidates=1000,
                                            unclip_ratio=1.6,
                                            use_dilation=True)

        sess_opt = onnxruntime.SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        self.session = onnxruntime.InferenceSession(det_model_path, sess_opt)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
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
        img_height, img_width = image_shape[0:2]
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

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        shape_list = np.expand_dims(shape_list, axis=0)

        starttime = time.time()
        preds = self.session.run([self.output_name],
                                 {self.input_name: img})

        post_result = self.postprocess_op(preds[0], shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, elapse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='image_path|image_dir')
    parser.add_argument('--model_path', type=str, help='model_path')
    args = parser.parse_args()

    text_detector = TextDetector(args.model_path)

    img, flag = check_and_read_gif(args.image_path)
    if not flag:
        img = cv2.imread(args.image_path)
    if img is None:
        raise ValueError(f"error in loading image:{args.image_path}")

    dt_boxes, elapse = text_detector(img)

    src_im = draw_text_det_res(dt_boxes, args.image_path)
    cv2.imwrite('det_results.jpg', src_im)
    print('图像已经保存为det_results.jpg了')
