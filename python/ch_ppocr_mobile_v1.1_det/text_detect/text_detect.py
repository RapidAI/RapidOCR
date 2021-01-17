# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: get_text_location.py
# @Time: 2021/01/16 11:56:04
# @Author: Max
from pathlib import Path
import onnxruntime
import time
import argparse

import cv2
from .utils import DBPostProcess, DBPreProcess, utils

root_path = Path(__file__).parent
det_model_path = str(root_path / 'weights' / 'ch_mobile_v1.1_det.onnx')


class TextDetector(object):
    def __init__(self):
        self.preprocess_op = DBPreProcess()
        self.session = onnxruntime.InferenceSession(det_model_path)
        self.postprocess_op = DBPostProcess()

    @staticmethod
    def load_image(image_path):
        img, flag = utils.check_and_read_gif(image_path)
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

        inputs = {self.session.get_inputs()[0].name: im}
        outputs = self.session.run(None, inputs)[0]

        dt_boxes = self.postprocess_op(outputs, [ratio_list], ori_im.shape)
        plot_img = utils.draw_text_det_res(dt_boxes, img)
        return dt_boxes, plot_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        default='images/1.jpg')
    args = parser.parse_args()

    text_detector = TextDetector()
    image_path = r'images/2021-01-16_11-39-35.png'
    dt_boxes, plot_img = text_detector(args.image_path)
    cv2.imwrite(f'det_results.jpg', plot_img)


