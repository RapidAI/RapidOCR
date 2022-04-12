# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: test_openvino.py
# @Author: SWHL
# @Contact: liekkaskono@163.com
import numpy as np
from openvino.runtime import Core

cls_model_path = r'models/ch_ppocr_mobile_v2.0_cls_infer.onnx'
ie = Core()
model_onnx = ie.read_model(cls_model_path)
compile_model = ie.compile_model(model_onnx, device_name='CPU')
pass

# Infer ONNXRuntime
# import cv2

# from ch_ppocr_mobile_v2_det import TextDetector

# det_model_path = 'models/ch_PP-OCRv2_det_infer.onnx'

# img_path = 'test_images/long1.jpg'
# img = cv2.imread(img_path)

# text_detect_onnx = TextDetector(det_model_path)
# dt_boxes, elapse = text_detect_onnx(img)
# print(f'onnxruntime: {elapse}')


# # Infer ONNX
# import cv2
# from ch_ppocr_mobile_v2_det_openvino_onnx import TextDetector as TextDetectorVino

# det_model_path = 'models/ch_PP-OCRv2_det_infer.onnx'
# img_path = 'test_images/det_images/ch_en_num.jpg'
# img = cv2.imread(img_path)

# text_detect_vino = TextDetectorVino(det_model_path)
# dt_boxes, elapse = text_detect_vino(img)
# print(f'openvino ONNX: {elapse}\tdt_boxes:{np.mean(dt_boxes)}')

# # # Infer IR
# import cv2
# from ch_ppocr_mobile_v2_det_openvino_ir import TextDetector as TextDetectorIR

# det_model_path = 'models/IR/static/ch_PP-OCRv2_det_infer.xml'
# img_path = 'test_images/long1.jpg'
# img = cv2.imread(img_path)

# text_detect_ir = TextDetectorIR(det_model_path)
# dt_boxes, elapse = text_detect_ir(img)
# print(f'FP16 openvino IR: {elapse}\tdt_boxes:{np.mean(dt_boxes)}')

# Infer paddle 有误
# RuntimeError: Can't SetBlob with name: x, because model input (shape={?,3,960,960}) and blob (shape=(1.3.12128.800)) are incompatible
# import cv2
# from ch_ppocr_mobile_v2_det_openvino_paddle import TextDetector as TextDetectorPaddle

# det_model_path = 'models/ch_PP-OCRv2_det_infer/ch_PP-OCRv2_det_infer/inference.pdmodel'
# img_path = 'test_images/long1.jpg'
# img = cv2.imread(img_path)

# text_detect_ir = TextDetectorPaddle(det_model_path)
# dt_boxes, elapse = text_detect_ir(img)
# print(f'openvino IR: {elapse}')
