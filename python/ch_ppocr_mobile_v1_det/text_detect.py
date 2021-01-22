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

    @staticmethod
    def load_image(image_path):
        img, flag = check_and_read_gif(image_path)
        if not flag:
            img = cv2.imread(image_path)
        if img is None:
            raise ValueError('图像有问题')
        return img

    def __call__(self, image_path):
        img = self.load_image(image_path)
        ori_im = img.copy()
        im, ratio_list = self.preprocess_op(img)
        if im is None:
            return None, 0
        starttime = time.time()
        inputs = {self.session.get_inputs()[0].name: im}
        outputs = self.session.run(None, inputs)[0]

        dt_boxes = self.postprocess_op(outputs, [ratio_list], ori_im.shape)
        # plot_img = draw_text_det_res(dt_boxes, img)
        elapse = time.time() - starttime
        return dt_boxes, elapse, ori_im


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str,
                        default='images/1.jpg', help='image_path|image_dir')
    parser.add_argument('model_path', type=str, default=None,
                        help='model_path')
    args = parser.parse_args()

    text_detector = TextDetector(args.model_path)
    dt_boxes, plot_img = text_detector(args.image_path)
    cv2.imwrite('det_results.jpg', plot_img)
    print('图像已经保存在det_results.jpg了')
