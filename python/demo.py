# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapidocr import RapidOCR

# model_dir = "/Users/joshuawang/projects/_self/PaddleOCRv5/models/official_models/PP-OCRv5_server_det"
# engine = RapidOCR(params={"Global.with_paddle": True, "Det.model_dir": model_dir})

engine_name = "paddle"
engine = RapidOCR(
    params={
        "Det.engine_name": engine_name,
        "Cls.engine_name": engine_name,
        "Rec.engine_name": engine_name,
    }
)
# engine = RapidOCR(params={"Cls.engine_name": "torch"})

img_url = "https://img1.baidu.com/it/u=3619974146,1266987475&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=516"
result = engine(img_url)
print(result)

result.vis("vis_result.jpg")
