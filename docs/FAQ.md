## 常见问题
#### Q: 请问这个能在32位C#中用嘛?
**A:** C#可以32位，要用32位的dll，但nuget上的onnxruntime不支持win7。

#### Q: Windows系统下，装完环境之后，运行示例程序之后，报错OSError: [WinError 126] 找不到指定的模組
**A:** 原因是Shapely库没有正确安装，如果是在Windows，可以在[Shapely whl](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)下载对应的whl包，离线安装即可；另外一种解决办法是用conda安装也可。

#### Q: Linux部署python的程序时，`import cv2`时会报`ImportError: ligGL.so.1: cannot open shared object file: No such file or directory`?
**A:** [解决方法](https://stackoverflow.com/a/63978454/3335415
) 有两个(来自群友ddeef)：
  1. 安装`opencv-python-headless`取代`opencv-python`;
  2. 运行`sudo apt-get install -y libgl1-mesa-dev`

#### Q: 询问下，我编译出来的进程在win7下面通过cmd调用，发生了崩溃的情况?
**A:** 不支持win7 (by @如果我有時光機)

#### Q: 能不能搞个openmmlab类似的那个提取信息的?
**A:** 这个目前正在调研测试当中，如果mmocr中关键信息提取效果还可以，后期会考虑整合进来。

#### Q: RapidOCR和PaddleOCR是什么关系呢？
**A:** RapidOCR是将PaddleOCR的预训练模型转为onnx模型，不依赖paddle框架，方便各个平台部署。

#### Q: onnxruntime arm32 有人编译过吗？ 我编译成功了，但是使用的时候libonnxruntime.so:-1: error: file not recognized: File format not recognized  应该是版本不匹配
**A:** 没遇到过。我是直接在当前平台编译的，我们用的是arm。估计是平台不兼容,建议在本身平台上编译。没遇到过问题。通常出在交叉编译方式下。

#### Q: 请问一下c++ demo必须要vs2017及以上版本吗?
**A:** 最好用vs2019

#### Q: 可以达到百度EasyEdge Free App的效果吗？
**A:** edge的模型应该没有开源。百度开源的模型里server det的识别效果可以达到，但是模型比较大。

#### Q: 我用c++推理onnx貌似是cpu推理的，gpu没有反应?
**A:** 如果想用GPU的话，需要安装onnxruntime-gpu版，自己在onnxruntime的代码中添加EP (execution provider)。我们的定位是通用，只用cpu推理。

#### Q: 您好，我想部署下咱们的ocr识别，有提供linux版本的ocr部署包吗?
**A:** linux版本的自己编译即可, 可以参考我们的action中的脚本；其实编译非常容易，安装个opencv后，在cmakelists.txt中修改一下onnxruntime的路径即可，具体参考这个： https://github.com/RapidOCR/RapidOCR/blob/main/.github/workflows/make-linux.yml

#### Q: onnxruntime编译好的C++库，哪里可以下载到？
**A:** 从这里：https://github.com/RapidOCR/OnnxruntimeBuilder/releases/tag/1.7.0

#### Q: 目前简单测试环境是  Win10 + Cygwin + gcc + 纯C编程，可以在C程序中直接接入简单OCR功能吗？
**A:** 直接使用API就行，API就是由c导出的

#### Q: 模型下载地址
**A:** [百度网盘](https://pan.baidu.com/s/1PTcgXG2zEgQU6A_A3kGJ3Q?pwd=jhai) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)

#### Q: onnxruntime 1.7 下出错：onnxruntime::SequentialExecutor::Execute] Non-zero status code returned while running ScatterND node. Name:'ScatterND@1' Status Message: updates
**A:** 由于模型只支持`onnxruntime=1.5.0`导致，请更新模型,下载地址见`Q3`

#### Q: 边缘总有一行文字无法识别，怎么办？
**A:** 在 padding 参数中添加一个值 ，默认是0,你可以添加5或10, 甚至更大，直到能识别为止。注意不要添加过大，会浪费内存。