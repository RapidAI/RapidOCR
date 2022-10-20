# RapidOcrOnnxCs

### 介绍
* 本项目为Windows平台C# WinForm范例。
* 依赖的包：Emgu.CV、MicroSoft.ML.OnnxRuntime、clipper_library

### Demo下载(win、mac、linux)
编译好的demo文件比较大，可以到Q群共享内下载

### 编译环境
1. Windows 10 x64
2. Visual Studio 2017或以上

### 编译说明
1. Vs2019打开RapidOcrOnnxCs.sln。
2. 解决方案资源管理器->OcrLib->右键->管理NuGet程序包->浏览->搜索->安装
* 注意：Emgu.CV要选作者是“Emgu Corporation”
* Emgu.CV 4.5.5.4823
* Emgu.CV.runtime.windows 4.5.5.4823
* MicroSoft.ML.OnnxRuntime 1.12.1
* clipper_library 6.2.1
3. 解决方案资源管理器->OcrLiteOnnxForm->右键->管理NuGet程序包->浏览->搜索->安装
* 注意：Emgu.CV要选作者是“Emgu Corporation”
* Emgu.CV 4.5.5.4823
* Emgu.CV.Bitmap 4.5.5.4823
4. 确保：OcrLiteOnnxForm设为启动项目
5. 确保：OcrLiteOnnxForm->右键->属性->生成->平台目标:x64
6. 确保：OcrLiteLib->右键->属性->生成->平台目标:x64
7. 生成解决方案
8. 把models文件夹复制到```\RapidOcrOnnxCs\OcrOnnxForm\bin\Debug(或Release)```
* [模型下载地址](https://github.com/znsoftm/BaiPiaoOCR/tree/main/models)
9. 运行

### 其它
* 修改模型路径，模型名称，线程数，必须“重新初始化”才能生效
* 输入参数说明请参考[BaiPiaoOcrOnnx项目](https://github.com/RapidAI/RapidOCR/blob/main/cpp/README.md)
