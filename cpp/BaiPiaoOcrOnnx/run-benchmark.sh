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

echo "请选择det模型: 1)server, 2)mobile"
read -p "" DET_MODEL
if [ $DET_MODEL == 1 ]; then
    DET_MODEL="ch_ppocr_server_v2.0_det_infer.onnx"
elif [ $DET_MODEL == 2 ]; then
    DET_MODEL="ch_ppocr_mobile_v2.0_det_infer.onnx"
else
  echo -e "Input Error!"
fi

echo "请输入循环次数:"
read -p "" LOOP_COUNT

TARGET_IMG=images/1.jpg
if [ ! -f "$TARGET_IMG" ]; then
echo "找不到待识别的目标图片：${TARGET_IMG}，请打开本文件并编辑TARGET_IMG"
exit
fi

##### run test on MacOS or Linux
./build/benchmark --version
./build/benchmark --models models \
--det $DET_MODEL \
--cls ch_ppocr_mobile_v2.0_cls_infer.onnx \
--rec ch_ppocr_server_v2.0_rec_infer.onnx \
--keys ppocr_keys_v1.txt \
--image $TARGET_IMG \
--numThread $NUM_THREADS \
--padding 0 \
--maxSideLen 1024 \
--boxScoreThresh 0.5 \
--boxThresh 0.3 \
--unClipRatio 1.5 \
--doAngle 1 \
--mostAngle 1 \
--loopCount $LOOP_COUNT