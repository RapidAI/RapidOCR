# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import time

import cv2

from rapidocr import TextSystem


root_path = "sdk_rapidocr_v1.0.0/"
det_model_path = "resources/models/ch_PP-OCRv2_det_infer.onnx"
cls_model_path = "resources/models/ch_ppocr_mobile_v2.0_cls_infer.onnx"
rec_model_path = "resources/models/ch_mobile_v2.0_rec_infer.onnx"
dict_path = "resources/ppocr_keys_v1.txt"

det_model_path = root_path + det_model_path
rec_model_path = root_path + rec_model_path
cls_model_path = root_path + cls_model_path
dict_path = root_path + dict_path

text_sys = TextSystem(det_model_path,
                      rec_model_path,
                      cls_model_path,
                      dict_path)

image_path = 'sdk_rapidocr_v1.0.0/images/1.jpg'
img = cv2.imread(image_path)

# rec_res:识别结果(结果，置信度)
# dt_boxes：对应文本框坐标
start_time = time.time()
dt_boxes, rec_res = text_sys(img)
txts = [rec_res[i][0] for i in range(len(rec_res))]

for txt in txts:
    print(txt)
