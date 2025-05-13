# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from rapidocr import RapidOCR

engine = RapidOCR()

# img_url = "https://img1.baidu.com/it/u=3619974146,1266987475&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=516"

# img_paths = ["tmp/en.jpg", "tmp/ch.jpg", "tmp/ch_en.jpg"]

# for img_path in img_paths:
#     result = engine(img_path, return_word_box=True)
#     # print(result)

#     result.vis(f"tmp/vis_{Path(img_path).stem}.jpg")
img_path = "tmp/ch.jpg"
result = engine(img_path, return_word_box=True)
result.vis(f"tmp/vis_{Path(img_path).stem}.jpg")
