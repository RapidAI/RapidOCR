# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2

from rapidocr_onnxruntime import RapidOCR, VisRes

# from rapidocr_paddle import RapidOCR, VisRes
# from rapidocr_openvino import RapidOCR, VisRes

engine = RapidOCR()
vis = VisRes()

image_path = "tests/test_files/ch_en_num.jpg"
with open(image_path, "rb") as f:
    img = f.read()

result, elapse_list = engine(img, return_word_box=True)
print(result)
print(elapse_list)

(boxes, txts, scores, words_boxes, words) = list(zip(*result))

font_path = "resources/fonts/FZYTK.TTF"
vis_img = vis(img, boxes, txts, scores, font_path)
cv2.imwrite("vis.png", vis_img)

words_boxes = sum(words_boxes, [])
words_all = sum(words, [])
words_scores = [1.0] * len(words_boxes)
vis_img = vis(img, words_boxes, words_all, words_scores, font_path)
cv2.imwrite("vis_single.png", vis_img)
