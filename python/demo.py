# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2

from rapidocr import RapidOCR, VisRes

# from rapidocr_onnxruntime import RapidOCR, VisRes


# from rapidocr_paddle import RapidOCR, VisRes
# from rapidocr_openvino import RapidOCR, VisRes

# yaml_path = "tests/test_files/config.yaml"
# engine = RapidOCR(config_path=yaml_path)

engine = RapidOCR(params={"Cls.model_path": "1.onnx"})
vis = VisRes()

image_path = "tests/test_files/ch_en_num.jpg"
with open(image_path, "rb") as f:
    img = f.read()

# result, elapse_list = engine(img, use_det=True, use_cls=False, use_rec=False)
result = engine(img, return_word_box=True)
print(result)
print(result.elapse)

boxes = result.boxes
txts = result.txts
scores = result.scores

font_path = "resources/fonts/FZYTK.TTF"
vis_img = vis(img, result.boxes, result.txts, result.scores, font_path)
cv2.imwrite("vis.png", vis_img)

words_results = result.word_results
words, words_scores, words_boxes = list(zip(*words_results))
vis_img = vis(img, words_boxes, words, words_scores, font_path)
cv2.imwrite("vis_single.png", vis_img)
