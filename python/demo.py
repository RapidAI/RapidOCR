# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2

from rapidocr_onnxruntime import RapidOCR, VisRes

# from rapidocr_openvino import RapidOCR, VisRes


rapid_ocr = RapidOCR()
vis = VisRes(font_path="resources/fonts/FZYTK.TTF")

image_path = "tests/test_files/ch_en_num.jpg"
with open(image_path, "rb") as f:
    img = f.read()

result, elapse_list = rapid_ocr(img)
print(result)
print(elapse_list)

boxes, txts, scores = list(zip(*result))
res = vis(img, boxes, txts, scores)
cv2.imwrite("vis.png", res)
