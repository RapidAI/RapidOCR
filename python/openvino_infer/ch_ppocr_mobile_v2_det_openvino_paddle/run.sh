#! /bin/bash
set -e errexit

root_path=$(cd $(dirname $0); pwd)
image_path=${root_path}"/test_images/det_images/1.jpg"
model_path=${root_path}"/models/ch_ppocr_mobile_v2.0_det_infer.onnx"

python text_detect.py --image_path ${image_path} \
                      --model_path ${model_path}