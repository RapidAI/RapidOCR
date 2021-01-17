# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: demo.py
# @Time: 2021/01/16 16:33:49
# @Author: Max
from text_detect import TextDetector
import cv2

test_detector = TextDetector()

image_path = 'images/1.jpg'

# dst_boxes: 检测到图像中的文本框坐标，ndarray格式
# (10, 4, 2)→[10个，4个坐标，每个坐标两个点]
dst_boxes, im = test_detector(image_path)

cv2.imwrite('images/det_results.jpg', im)