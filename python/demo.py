# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapidocr import RapidOCR

engine = RapidOCR()

img_url = "https://github.com/RapidAI/RapidOCR/blob/main/python/tests/test_files/ch_en_num.jpg?raw=true"
img_url = "tmp/1.png"
result = engine(img_url, return_word_box=True)
print(result)

result.vis("vis_result.jpg")
