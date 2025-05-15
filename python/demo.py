# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapidocr import RapidOCR

# model_dir = "/Users/joshuawang/projects/_self/tmp/RapidOCR/paddle/PP-OCRv4/rec/ch_doc_PP-OCRv4_rec_server_infer"
# engine = RapidOCR(
#     params={
#         "Global.lang_rec": "ch_doc_server",
#         "Global.with_paddle": True,
#         "Rec.model_dir": model_dir,
#     }
# )

engine = RapidOCR(
    params={"Global.lang_rec": "ch_doc_server", "Global.with_paddle": True}
)

# img_url = "https://img1.baidu.com/it/u=3619974146,1266987475&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=516"
# result = engine(img_url)

img_path = "tests/test_files/ch_doc_server.png"
result = engine(img_path)
print(result)

result.vis("vis_result.jpg")
