#! /bin/bash
set -e errexit

root_path=$(cd $(dirname $0); pwd)
image_path=${root_path}"/test_images/rec_images/2021-01-18_14-22-32.png"
model_path=${root_path}"/models/ch_ppocr_mobile_v2.0_cls_infer.onnx"

python text_cls.py --image_path ${image_path} \
                   --model_path ${model_path}