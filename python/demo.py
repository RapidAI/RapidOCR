# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapidocr import RapidOCR

engine = RapidOCR()

# img_url = "https://img1.baidu.com/it/u=3619974146,1266987475&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=516"
img_path = "debug.jpg"
img_path = "ch_en.jpg"
# img_path = "tests/test_files/ch_en_num.jpg"
result = engine(img_path, return_word_box=True)
print(result)

result.vis("vis_result.jpg")
