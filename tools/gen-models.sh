#!/bin/bash

# ======ch_ppocr_mobile_v2.0_cls======
python3 tools/export_custom.py --height 48 --width 192 -c configs/cls/cls_mv3.yml -o Global.pretrained_model=./models/ch_ppocr_mobile_v2.0_cls_train/best_accuracy Global.load_static_weights=False Global.save_inference_dir=./models/ch_ppocr_mobile_v2.0_cls_infer/

paddle2onnx --model_dir models/ch_ppocr_mobile_v2.0_cls_infer/  --model_filename  inference.pdmodel --params_filename inference.pdiparams --save_file models/ch_ppocr_mobile_v2.0_cls_infer.onnx --opset_version 11 --enable_onnx_checker True

# ======ch_ppocr_mobile_v2.0_det======
python3 tools/export_custom.py --height -1 --width -1 -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml -o Global.pretrained_model=./models/ch_ppocr_mobile_v2.0_det_train/best_accuracy Global.load_static_weights=False Global.save_inference_dir=./models/ch_ppocr_mobile_v2.0_det_infer/

paddle2onnx --model_dir models/ch_ppocr_mobile_v2.0_det_infer/  --model_filename  inference.pdmodel --params_filename inference.pdiparams --save_file models/ch_ppocr_mobile_v2.0_det_infer.onnx --opset_version 11 --enable_onnx_checker True

# ======ch_ppocr_mobile_v2.0_rec======
python3 tools/export_custom.py --height 32 --width -1 -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml -o Global.pretrained_model=./models/ch_ppocr_mobile_v2.0_rec_train/best_accuracy Global.load_static_weights=False Global.save_inference_dir=./models/ch_ppocr_mobile_v2.0_rec_infer/

paddle2onnx --model_dir models/ch_ppocr_mobile_v2.0_rec_infer/  --model_filename  inference.pdmodel --params_filename inference.pdiparams --save_file models/ch_ppocr_mobile_v2.0_rec_infer.onnx --opset_version 11 --enable_onnx_checker True


# ======ch_ppocr_server_v2.0_det======
python3 tools/export_custom.py --height -1 --width -1 -c configs/det/ch_ppocr_v2.0/ch_det_res18_db_v2.0.yml -o Global.pretrained_model=./models/ch_ppocr_server_v2.0_det_train/best_accuracy Global.load_static_weights=False Global.save_inference_dir=./models/ch_ppocr_server_v2.0_det_infer/

paddle2onnx --model_dir models/ch_ppocr_server_v2.0_det_infer/  --model_filename  inference.pdmodel --params_filename inference.pdiparams --save_file models/ch_ppocr_server_v2.0_det_infer.onnx --opset_version 11 --enable_onnx_checker True


# ======ch_ppocr_server_v2.0_rec======
python3 tools/export_custom.py --height 32 --width -1 -c configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml -o Global.pretrained_model=./models/ch_ppocr_server_v2.0_rec_train/best_accuracy Global.load_static_weights=False Global.save_inference_dir=./models/ch_ppocr_server_v2.0_rec_infer/

paddle2onnx --model_dir models/ch_ppocr_server_v2.0_rec_infer/  --model_filename  inference.pdmodel --params_filename inference.pdiparams --save_file models/ch_ppocr_server_v2.0_rec_infer.onnx --opset_version 11 --enable_onnx_checker True




