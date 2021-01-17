chcp 65001
:: Set Param
@ECHO OFF
@SETLOCAL
echo "Setting the Number of Threads=%NUMBER_OF_PROCESSORS% Using an OpenMP Environment Variable"
set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%

echo "请选择det模型: 1)server, 2)mobile"
set /p flag=
if %flag% == 1 (set DET_MODEL=ch_ppocr_server_v2.0_det_infer.onnx)^
else if %flag% == 2 (set DET_MODEL=ch_ppocr_mobile_v2.0_det_infer.onnx)^
else (echo 输入错误！Input Error!)

SET TARGET_IMG=images/1.jpg
if not exist %TARGET_IMG% (
echo "找不到待识别的目标图片：%TARGET_IMG%，请打开本文件并编辑TARGET_IMG"
PAUSE
exit
)

:: run Windows
build\BaiPiaoOcrOnnx.exe --version
build\BaiPiaoOcrOnnx.exe --models models ^
--det %DET_MODEL% ^
--cls ch_ppocr_mobile_v2.0_cls_infer.onnx ^
--rec ch_ppocr_server_v2.0_rec_infer.onnx ^
--keys ppocr_keys_v1.txt ^
--image %TARGET_IMG% ^
--numThread %NUMBER_OF_PROCESSORS% ^
--padding 0 ^
--maxSideLen 1024 ^
--boxScoreThresh 0.5 ^
--boxThresh 0.3 ^
--unClipRatio 1.5 ^
--doAngle 1 ^
--mostAngle 1
PAUSE
@ENDLOCAL
