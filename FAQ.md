## 常见问题

#### Q9: 请问一下c++ demo必须要vs2017及以上版本吗?
**A:** 最好用vs2019

#### Q8: 可以达到百度EasyEdge Free App的效果吗？
**A:** edge的模型应该没有开源。百度开源的模型里server det的识别效果可以达到，但是模型比较大。

#### Q7: 我用c++推理onnx貌似是cpu推理的，gpu没有反应?
**A:** 如果想用GPU的话，需要安装onnxruntime-gpu版,自己在onnxruntime的代码中添加EP (execution provider)。我们的定位是通用，只用cpu推理。

#### Q6: 您好，我想部署下咱们的ocr识别，有提供linux版本的ocr部署包吗?
**A:** linux版本的自己编译即可, 可以参考我们的action中的脚本；其实编译非常容易，安装个opencv后，在cmakelists.txt中修改一下onnxruntime的路径即可，具体参考这个： https://github.com/RapidOCR/RapidOCR/blob/main/.github/workflows/make-linux.yml

#### Q5: onnxruntime编译好的C++库，哪里可以下载到？
**A:** 从这里：https://github.com/RapidOCR/OnnxruntimeBuilder/releases/tag/1.7.0

#### Q4: 目前简单测试环境是  Win10 + Cygwin + gcc + 纯C编程，可以在C程序中直接接入简单OCR功能吗？
**A:** 直接使用API就行，API就是由c导出的

#### Q1: 边缘总有一行文字无法识别，怎么办？

**A:** 在 padding 参数中添加一个值 ，默认是0,你可以添加5或10, 甚至更大，直到能识别为止。注意不要添加过大，会浪费内存。

#### Q2: onnxruntime 1.7 下出错：onnxruntime::SequentialExecutor::Execute] Non-zero status code returned while running ScatterND node. Name:'ScatterND@1' Status Message: updates

**A:** 由于模型只支持`onnxruntime=1.5.0`导致，请更新模型,下载地址见`Q3`

#### Q3: 模型下载地址

**A:** [提取码：30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)
