# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: get_text_location.py
# @Time: 2021/01/16 11:56:04
# @Author: Max
import argparse
import time

import cv2
import onnxruntime

try:
    from .utils import (DBPostProcess, DBPreProcess, check_and_read_gif,
                        draw_text_det_res)
except:
    from utils import (DBPostProcess, DBPreProcess, check_and_read_gif,
                       draw_text_det_res)


class TextDetector(object):
    def __init__(self, det_model_path):
        self.preprocess_op = DBPreProcess()
        self.session = onnxruntime.InferenceSession(det_model_path)
        self.postprocess_op = DBPostProcess()

    def __call__(self, img):
        ori_im = img.copy()
        im, ratio_list = self.preprocess_op(img)
        if im is None:
            return None, 0
        starttime = time.time()
        inputs = {self.session.get_inputs()[0].name: im}
        outputs = self.session.run(None, inputs)[0]

        dt_boxes = self.postprocess_op(outputs, [ratio_list], ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, elapse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str,
                        default=None, help='image_path|image_dir')
    parser.add_argument('model_path', type=str, default=None,
                        help='model_path')
    args = parser.parse_args()

    text_detector = TextDetector(args.model_path)

    img, flag = check_and_read_gif(args.image_path)
    if not flag:
        img = cv2.imread(args.image_path)
    if img is None:
        raise ValueError(f"error in loading image:{args.image_path}")

    dt_boxes, elapse = text_detector(img)

    plot_img = draw_text_det_res(dt_boxes, img)
    cv2.imwrite('det_results.jpg', plot_img)
    print('图像已经保存为det_results.jpg了')
