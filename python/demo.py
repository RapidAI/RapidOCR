# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2

from rapidocr import RapidOCR, VisRes

engine = RapidOCR(params={"Global.with_openvino": True})
vis = VisRes()

image_path = "tests/test_files/ch_en_num.jpg"
with open(image_path, "rb") as f:
    img = f.read()

result = engine(img, return_word_box=True)
print(result)
print(result.elapse)

boxes = result.boxes
txts = result.txts
scores = result.scores

vis_img = vis(img, result.boxes, result.txts, result.scores)
cv2.imwrite("vis.png", vis_img)

words_results = result.word_results
words, words_scores, words_boxes = list(zip(*words_results))
vis_img = vis(img, words_boxes, words, words_scores)
cv2.imwrite("vis_single.png", vis_img)
