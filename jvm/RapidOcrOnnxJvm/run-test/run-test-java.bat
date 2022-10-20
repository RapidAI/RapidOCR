chcp 65001
:: Set Param
@ECHO OFF
@SETLOCAL
echo "Setting the Number of Threads=%NUMBER_OF_PROCESSORS% Using an OpenMP Environment Variable"
set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%

SET TARGET_IMG=images/1.jpg
if not exist %TARGET_IMG% (
echo "找不到待识别的目标图片：%TARGET_IMG%，请打开本文件并编辑TARGET_IMG"
PAUSE
exit
)

:: run Windows
java -Djava.library.path=. -Dfile.encoding=UTF-8 -jar BaiPiaoOcrOnnxJvm.jar models ^
ch_ppocr_server_v2.0_det_infer.onnx \
ch_ppocr_mobile_v2.0_cls_infer.onnx \
ch_ppocr_server_v2.0_rec_infer.onnx \
ppocr_keys_v1.txt \
%TARGET_IMG% \
%NUMBER_OF_PROCESSORS% \
50 \
1024 \
0.5 \
0.3 \
1.6 \
1 \
0

::models
::det
::cls
::rec
::keys
::image
::numThread
::padding
::maxSideLen
::boxScoreThresh
::boxThresh
::unClipRatio
::doAngle
::mostAngle

PAUSE
@ENDLOCAL
