#!/bin/bash
# convert paddleocr to onnx
 paddle2onnx --model_dir models/ch_ppocr_mobile_v2.0_cls_infer/  --model_filename  inference.pdmodel --params_filename inference.pdiparams --save_file ch_ppocr_mobile_v2.0_cls_infer.onnx --opset_version 11 --enable_onnx_checker True
 paddle2onnx --model_dir models/ch_ppocr_mobile_v2.0_det_infer/  --model_filename  inference.pdmodel --params_filename inference.pdiparams --save_file ch_ppocr_mobile_v2.0_det_infer.onnx --opset_version 11 --enable_onnx_checker True
 paddle2onnx --model_dir models/ch_ppocr_mobile_v2.0_rec_infer/  --model_filename  inference.pdmodel --params_filename inference.pdiparams --save_file ch_ppocr_mobile_v2.0_rec_infer.onnx   --opset_version 11 --enable_onnx_checker True
