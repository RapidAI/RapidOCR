# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2

from rapidocr_onnxruntime import RapidOCR, VisRes

# from rapidocr_paddle import RapidOCR, VisRes
# from rapidocr_openvino import RapidOCR, VisRes


engine = RapidOCR()
vis = VisRes()

image_path = "tests/test_files/black_font_color_transparent.png"
with open(image_path, "rb") as f:
    img = f.read()

result, elapse_list = engine(img)
print(result)
print(elapse_list)

boxes, txts, scores = list(zip(*result))

font_path = "resources/fonts/FZYTK.TTF"
vis_img = vis(img, boxes, txts, scores, font_path)
cv2.imwrite("vis.png", vis_img)
