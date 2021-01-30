#!/bin/bash

function trainToOnnx() {
  python3 tools/export_custom.py --height ${1} \
  --width ${2} \
  -c ${3} \
  -o Global.pretrained_model=./inference/${4}_train/best_accuracy Global.load_static_weights=False Global.save_inference_dir=./inference/${4}_infer/

  paddle2onnx --model_dir inference/${4}_infer  \
   --model_filename  inference.pdmodel \
   --params_filename inference.pdiparams \
   --save_file inference/${4}_infer.onnx \
   --opset_version 11 \
   --enable_onnx_checker True
}

# ch_ppocr_mobile_v2.0_cls
trainToOnnx 48 192 configs/cls/cls_mv3.yml ch_ppocr_mobile_v2.0_cls

# ch_ppocr_mobile_v2.0_det
trainToOnnx -1 -1 configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml ch_ppocr_mobile_v2.0_det

# ch_ppocr_server_v2.0_det
trainToOnnx -1 -1 configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml ch_ppocr_server_v2.0_det

# ch_ppocr_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml ch_ppocr_mobile_v2.0_rec

# ch_ppocr_server_v2.0_rec
trainToOnnx 32 -1 configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml ch_ppocr_server_v2.0_rec

# en_number_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_en_number_lite_train.yml en_number_mobile_v2.0_rec

# french_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_french_lite_train.yml french_mobile_v2.0_rec

# german_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_german_lite_train.yml german_mobile_v2.0_rec

# korean_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_korean_lite_train.yml korean_mobile_v2.0_rec

# japan_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_japan_lite_train.yml japan_mobile_v2.0_rec

# it_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_it_lite_train.yml it_mobile_v2.0_rec

# xi_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_xi_lite_train.yml xi_mobile_v2.0_rec

# pu_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_pu_lite_train.yml pu_mobile_v2.0_rec

# ru_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_ru_lite_train.yml ru_mobile_v2.0_rec

# ar_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_ar_lite_train.yml ar_mobile_v2.0_rec

# hi_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_hi_lite_train.yml hi_mobile_v2.0_rec

# chinese_cht_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_chinese_cht_lite_train.yml chinese_cht_mobile_v2.0_rec

# ug_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_ug_lite_train.yml ug_mobile_v2.0_rec

# fa_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_fa_lite_train.yml fa_mobile_v2.0_rec

# ur_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_ur_lite_train.yml ur_mobile_v2.0_rec

# rs_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_rs_lite_train.yml rs_mobile_v2.0_rec

# oc_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_oc_lite_train.yml oc_mobile_v2.0_rec

# mr_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_mr_lite_train.yml mr_mobile_v2.0_rec

# ne_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_ne_lite_train.yml ne_mobile_v2.0_rec

# rsc_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_rsc_lite_train.yml rsc_mobile_v2.0_rec

# bg_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_bg_lite_train.yml bg_mobile_v2.0_rec

# uk_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_uk_lite_train.yml uk_mobile_v2.0_rec

# be_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_be_lite_train.yml be_mobile_v2.0_rec

# te_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_te_lite_train.yml te_mobile_v2.0_rec

# ka_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_ka_lite_train.yml ka_mobile_v2.0_rec

# ta_mobile_v2.0_rec
trainToOnnx 32 -1 configs/rec/multi_language/rec_ta_lite_train.yml ta_mobile_v2.0_rec


