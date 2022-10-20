#!/bin/bash
## script for 内存泄露检查
# ========== macOS ==========
# https://github.com/LouisBrunner/valgrind-macos
# brew tap LouisBrunner/valgrind
# brew install --HEAD LouisBrunner/valgrind/valgrind
# ========== linux ==========
# https://www.valgrind.org/
# apt install valgrind

NUM_THREADS=1
echo "Setting the Number of Threads=$NUM_THREADS Using an OpenMP Environment Variable"
set OMP_NUM_THREADS=$NUM_THREADS

##### run test on MacOS or Linux
valgrind --trace-children=yes --tool=memcheck --leak-check=full --leak-resolution=med --track-origins=yes --vgdb=no --log-file=valgrind-memcheck.txt \
java -Djava.library.path=. -jar BaiPiaoOcrOnnxJvm.jar models \
ch_ppocr_server_v2.0_det_infer.onnx \
ch_ppocr_mobile_v2.0_cls_infer.onnx \
ch_ppocr_server_v2.0_rec_infer.onnx \
ppocr_keys_v1.txt \
images/1.jpg \
$NUM_THREADS \
0 \
1024 \
0.5 \
0.3 \
1.6 \
1 \
0

#models
#det
#cls
#rec
#keys
#image
#numThread
#padding
#maxSideLen
#boxScoreThresh
#boxThresh
#unClipRatio
#doAngle
#mostAngle