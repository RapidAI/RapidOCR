#!/usr/bin/env bash

sysOS=`uname -s`
NUM_THREADS=1
if [ $sysOS == "Darwin" ];then
    #echo "I'm MacOS"
    NUM_THREADS=$(sysctl -n hw.ncpu)
elif [ $sysOS == "Linux" ];then
    #echo "I'm Linux"
    NUM_THREADS=$(grep ^processor /proc/cpuinfo | wc -l)
else
    echo "Other OS: $sysOS"
fi

echo "Setting the Number of Threads=$NUM_THREADS Using an OpenMP Environment Variable"
set OMP_NUM_THREADS=$NUM_THREADS

##### run test on MacOS or Linux
java -Djava.library.path=. -jar BaiPiaoOcrOnnxJvm.jar models \
ch_ppocr_server_v2.0_det_infer.onnx \
ch_ppocr_mobile_v2.0_cls_infer.onnx \
ch_ppocr_server_v2.0_rec_infer.onnx \
ppocr_keys_v1.txt \
images/1.jpg \
$NUM_THREADS \
50 \
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