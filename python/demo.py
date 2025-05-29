# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapidocr import EngineType, RapidOCR

# engine = RapidOCR(
#     params={
#         "Det.ocr_version": OCRVersion.PPOCRV5,
#         "Det.engine_type": EngineType.PADDLE,
#         "Det.model_type": ModelType.SERVER,
#     }
# )

# img_url = "https://img1.baidu.com/it/u=3619974146,1266987475&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=516"
# result = engine(img_url)
# print(result)

# result.vis("vis_result.jpg")


engine = RapidOCR(
    params={
        "Det.engine_type": EngineType.OPENVINO,
        "Cls.engine_type": EngineType.OPENVINO,
        "Rec.engine_type": EngineType.OPENVINO,
    }
)

img_url = "https://github.com/RapidAI/RapidOCR/blob/main/python/tests/test_files/ch_en_num.jpg?raw=true"
result = engine(img_url)
print(result)

result.vis("vis_result.jpg")
