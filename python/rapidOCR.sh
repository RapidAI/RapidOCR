#! /bin/bash

# The img_path can be file path or folder path.
img_path="test_images/det_images/1.jpg"

python rapidOCR.py --det_model_path models/ch_ppocr_mobile_v2.0_det_infer.onnx \
                   --cls_model_path models/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                   --rec_model_path models/ch_ppocr_mobile_v2.0_rec_infer.onnx \
                   --image_path ${img_path} \
                   --text_score 0.5
