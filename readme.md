完全开源免费并支持离线部署的多平台多语言OCR SDK

中文广告： 欢迎加入我们的QQ群下载模型及测试程序，qq群号：887298230

欢迎STAR, 欢迎PR, 欢迎转发。

## SDK 编译状态

| 平台            | 编译状态 |   提供状态 |
| --------------- | -------- | -------- |
| Windows x86/x64 |  [![CMake-windows-x86-x64](https://github.com/RapidOCR/RapidOCR/actions/workflows/windows-all-build.yaml/badge.svg)](https://github.com/RapidOCR/RapidOCR/actions/workflows/windows-all-build.yaml)        |  右侧下载 |
| Linux x64       |  [![CMake-linux](https://github.com/RapidOCR/RapidOCR/actions/workflows/make-linux.yml/badge.svg)](https://github.com/RapidOCR/RapidOCR/actions/workflows/make-linux.yml) |  暂不提供，自行编译 |


## 常见问题
 [FAQ](FAQ.md)

# 重要说明 

新模型已经完全兼容ONNXRuntime 1.7 或更高版本。 特别感谢：@Channingss

新版onnxruntime比1.6.0 性能提升40%以上。

## 在线演示(online demo)：  http://rapidocr.51pda.cn:9003/

### 演示模型组合说明： server det+mobile cls+mobile rec

中文详细说明，请访问： [中文版](readme_cn.md)           Chinese Name: 捷智OCR

### 本仓库包括


- [x]  纯c／c＋＋ API 接口，方便移植所有平台 （网盘提供预编译SDK）
- [x] .NET测试程序　　演示在.NET程序中的使用方法
- [x]  Android 测试程序
- [x] Java 测试程序
- [x]  python 测试程序
- [x]  WEB 演示   基于web的演示代码
- [ ]  IOS演示   招贡献者
- [ ]  依据python版本重写C++推理代码，以提升推理效果，并增加对gif/tga/webp 格式图片的支持
### 下载转换好的多语言模型：

```
Download OCR　models

https://github.com/RapidOCR/RapidOCR/releases/download/V1.0/rapid-model.tgz

```

----------------  For  English Users ----------------------

License:  Apache License 2.0

Note:

For PC (political correctness ), we changed our name from BaiPiaoOCR to RapidOCR


Special thanks to Channingss@baidu (the owner of project Paddle2Onnx)

### Directory structure

### [Click to see 目录结构](dir.md)    


**Copyright announcement:**

If you use or reference code or code snippet from the repository, please add our url 

https://github.com/RapidOCR/RapidOCR  in your prodouct derived from the repository.


**Author:**

benjaminwan, znsoftm

All contributors is in the contributor list on the right side of this page.


**Description**

BaiPiaoOCR means an  OCR Engine who is fair-skinned, rich and beautiful from Baidu, it's based on PaddleOCR & OnnxRuntime.

Note： The project is derived from https://github.com/PaddlePaddle/PaddleOCR

We create the project to transfer PaddleOCR's model into the version of Onnx model to deploy the model on almost every devices, such as x86 PC, Android & IOS.

We are using Paddle2ONNX (https://github.com/PaddlePaddle/Paddle2ONNX)  to transform it to Onnx format.




You can visualize the model by https://github.com/lutzroeder/netron/


Welcome to our Group on QQ: 887298230
or visit: https://jq.qq.com/?_wv=1027&k=P9b3olx6 to join us.


Download OCR　models
```
Linkage ：https://github.com/RapidOCR/RapidOCR/releases/download/V1.0/rapid-model.tg
```

### Demonstration with C++/JVM

![avatar](test_imgs/test_cpp.png)

### Demonstration with  .Net

![avatar](test_imgs/test_cs.png)
