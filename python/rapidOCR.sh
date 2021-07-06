#! /bin/bash
set -e errexit

python rapidOCR.py --det_model_path models/ch_ppocr_mobile_v2.0_det_infer.onnx \
                   --cls_model_path models/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                   --rec_model_path models/ch_ppocr_mobile_v2.0_rec_infer.onnx \
                   --image_path test_images/det_images/1.jpg
