#! /bin/bash

det_model_path="models/ch_PP-OCRv2_det_infer.onnx"
cls_model_path="models/ch_ppocr_mobile_v2.0_cls_infer.onnx"

# 这里可以按照自己需求，做相关更改
rec_model_path="models/ch_ppocr_mobile_v2.0_rec_infer.onnx"
keys_path="ch_ppocr_mobile_v2_rec/ppocr_keys_v1.txt"

img_path="test_images/det_images/ch_en_num.jpg"

python rapid_ocr_api.py --det_model_path ${det_model_path} \
                        --cls_model_path ${cls_model_path} \
                        --rec_model_path ${rec_model_path} \
                        --image_path ${img_path} \
                        --keys_path ${keys_path}
