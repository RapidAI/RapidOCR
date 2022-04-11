# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapid_ocr_api import TextSystem, visualize

det_model_path = 'models/ch_ppocr_mobile_v2.0_det_infer.onnx'
cls_model_path = 'models/ch_ppocr_mobile_v2.0_cls_infer.onnx'

# 中英文识别
rec_model_path = 'models/ch_ppocr_mobile_v2.0_rec_infer.onnx'
keys_path = 'rec_dict/ppocr_keys_v1.txt'

text_sys = TextSystem(det_model_path,
                      rec_model_path,
                      use_angle_cls=True,
                      cls_model_path=cls_model_path,
                      keys_path=keys_path)

image_path = r'test_images/det_images/ch_en_num.jpg'
dt_boxes, rec_res = text_sys(image_path)
visualize(image_path, dt_boxes, rec_res)


# # 只有中英文和数字识别
# rec_model_path = 'models/en_number_mobile_v2.0_rec_infer.onnx'
# keys_path = 'rec_dict/en_dict.txt'

# text_sys = TextSystem(det_model_path,
#                       rec_model_path,
#                       use_angle_cls=True,
#                       cls_model_path=cls_model_path,
#                       keys_path=keys_path)

# image_path = r'test_images/det_images/en_num.png'
# dt_boxes, rec_res = text_sys(image_path)
# visualize(image_path, dt_boxes, rec_res)


# # 日语识别
# rec_model_path = 'models/japan_rec_crnn.onnx'
# keys_path = 'rec_dict/japan_dict.txt'

# text_sys = TextSystem(det_model_path,
#                       rec_model_path,
#                       use_angle_cls=True,
#                       cls_model_path=cls_model_path,
#                       keys_path=keys_path)

# image_path = r'test_images/det_images/japan.png'
# dt_boxes, rec_res = text_sys(image_path)
# visualize(image_path, dt_boxes, rec_res)


# # 韩语识别
# rec_model_path = 'models/korean_mobile_v2.0_rec_infer.onnx'
# keys_path = 'rec_dict/korean_dict.txt'
# font_path = 'fonts/korean.ttf'

# text_sys = TextSystem(det_model_path,
#                       rec_model_path,
#                       use_angle_cls=True,
#                       cls_model_path=cls_model_path,
#                       keys_path=keys_path)

# image_path = r'test_images/det_images/korean_1.jpg'
# dt_boxes, rec_res = text_sys(image_path)
# visualize(image_path, dt_boxes, rec_res, font_path)
