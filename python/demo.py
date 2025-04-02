# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

from rapidocr import RapidOCR

engine = RapidOCR()

# img_url = "https://img1.baidu.com/it/u=3619974146,1266987475&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=516"
# result = engine(img_url)
# print(result)

# result.vis("vis_result.jpg")
img_list = list(
    Path(
        "/Users/joshuawang/projects/SourceCode/PaddleOCR-release-2.7/doc/imgs"
    ).iterdir()
)
for img_path in img_list:
    result = engine(img_path)
    print("----")
