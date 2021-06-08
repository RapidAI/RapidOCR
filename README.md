
# RapidOCR (捷智OCR)

简体中文 | [English](README_en.md)

## 简介
- 完全开源免费并支持离线部署的多平台多语言OCR SDK
- **中文广告**： 欢迎加入我们的QQ群下载模型及测试程序，qq群号：887298230
- **缘起**：百度paddlepaddle工程化不是太好，为了方便大家在各种端上进行ocr推理，我们将它转换为onnx格式，使用``python/c++/java/swift/c#`` 将它移植到各个平台。

- **名称来源**： 轻快好省并智能。 基于深度学习技术的OCR技术，主打人工智能优势及小模型，以速度为使命，效果为主导。

- 基于百度的开源PaddleOCR 模型及训练，任何人可以使用本推理库，也可以根据自己的需求使用百度的paddlepaddle 框架进行模型优化。

## 近期更新
#### 2021-06-08 update
- 整理仓库，统一模型下载路径
- 完善相关说明文档

#### 2021-04-18 update
- 新模型已经完全兼容ONNXRuntime 1.7 或更高版本。 特别感谢：@Channingss
- 新版onnxruntime比1.6.0 性能提升40%以上。

## 常见问题  [FAQ](FAQ.md)

## SDK 编译状态
鉴于ubuntu用户都是商业用户，也有编译能力，暂不提供预编译包使用，可自行编译。

| 平台            | 编译状态 |   提供状态 |
| --------------- | -------- | -------- |
| Windows x86/x64 |  [![CMake-windows-x86-x64](https://github.com/RapidOCR/RapidOCR/actions/workflows/windows-all-build.yaml/badge.svg)](https://github.com/RapidOCR/RapidOCR/actions/workflows/windows-all-build.yaml)        |  右侧下载 |
| Linux x64       |  [![CMake-linux](https://github.com/RapidOCR/RapidOCR/actions/workflows/make-linux.yml/badge.svg)](https://github.com/RapidOCR/RapidOCR/actions/workflows/make-linux.yml) |  暂不提供，自行编译 |

## 在线demo
- [Web demo](http://rapidocr.51pda.cn:9003/)
- demo所用模型组合为: **server det** + **mobile cls** + **mobile rec**

### [模型训练相关问题](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/FAQ.md)


### 目录结构
<details>
    <summary>(点击展开)</summary>

    RapidOCR
    ├── android             # 安卓工程目录
    ├── api4cpp             # c语言跨平台接口库源码目录，直接用根下的CMakelists.txt 编译
    ├── assets              # 一些演示用的图片，不是测试集
    ├── commonlib           # 通用库
    ├── cpp                 # 基于c++的工程项目文件夹
    ├── datasets            # 额外的训练数据集，百度网盘下载地址
    ├── dotnet              # .Net程序目录
    ├── FAQ.md              # 一些问答整理
    ├── images              # 测试用图片，两张典型的测试图，一张是自然场景，另一个为长文本
    ├── include             # 编译c语言接口库时的头文件目录
    ├── ios                 # 苹果手机平台工程目录
    ├── jvm                 # 基于java的工程目录
    ├── lib                 # 编译用库文件目录，用于编译c语言接口库用，默认并不上传二进制文件
    ├── models              # 放置可使用的模型文件下载信息，基于百度网盘
    ├── ocrweb              # 基于python和Flask web
    ├── python              # python推理代码目录
    ├── release             #
    ├── tools               #  一些转换脚本之类
    └── training            # 训练自己的模型使用的脚本或程序

</details>


### 当前进展
- [x] C++范例(Windows/Linux/macOS): [demo](./cpp)
- [x] Jvm范例(Java/Kotlin): [demo](./jvm)
- [x] .Net范例(C#): [demo](./dotnet)
- [x] Android范例: [demo](./android)
- [x] python范例: [demo](./python)
- [ ] IOS范例: 等待有缘人贡献代码
- [ ] 依据python版本重写C++推理代码，以提升推理效果，并增加对gif/tga/webp 格式图片的支持

### 模型转换
- 目前转换脚本支持的模型：
    - 文本方向分类模型1个，
    - 检测模型2个，
    - 识别模型28个(简体中文2种，繁体中文等26种)，合计31个
- [模型转换说明](./models)

### onnx模型下载
- [提取码：30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)

### 原始发起者及初创作者
- [benjaminwan](https://github.com/benjaminwan)
- [znsoftm](https://github.com/znsoftm)

### 版权声明
- 如果你的产品使用了本仓库中的全部或部分代码、文字或材料
- 请注明出处并包括我们的github url: `https://github.com/RapidOCR/RapidOCR`

### 授权
- OCR模型版权归百度所有，其它工程代码版权归本仓库所有者所有。
- 本软件采用LGPL 授权方式，欢迎大家贡献代码，提交issue 甚至pr.

### 联系我们
- 您可以通过QQ群联系到我们：**887298230**
- 群号搜索不到时，请直接点此[**链接**](https://jq.qq.com/?_wv=1027&k=P9b3olx6)，找到组织
- 用QQ扫描以下二维码:
    <p>
        <img src="./assets/qq_team.bmp" width="30%" height="30%" align="center">
    </p>
