
<div align="center">
  <img src="./assets/RapidOCR_LOGO.png" width="50%" height="50%"/>
</div>

# RapidOCR (捷智OCR)

简体中文 | [English](./docs/README_en.md)

<p align="left">
    <a href="https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/assets/RapidOCRDemo.ipynb" target="_blank"><img src="./assets/colab-badge.svg" alt="Open in Colab"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python-3.6+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors"><img src="https://img.shields.io/github/contributors/RapidAI/RapidOCR?color=9ea"></a>
    <a href="https://github.com/RapidAI/RapidOCR/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidOCR?color=ccf"></a>
</p>


<details>
    <summary>目录</summary>

- [RapidOCR (捷智OCR)](#rapidocr-捷智ocr)
  - [简介](#简介)
  - [近期更新(more)](#近期更新more)
      - [📌2022-06-25 update:](#2022-06-25-update)
      - [🥟2022-05-25 update](#2022-05-25-update)
      - [🍿2022-05-15 update](#2022-05-15-update)
      - [😀2022-05-12 upadte](#2022-05-12-upadte)
      - [🎧2022-04-04 update](#2022-04-04-update)
  - [整个框架](#整个框架)
  - [常见问题  FAQ](#常见问题--faq)
  - [SDK 编译状态](#sdk-编译状态)
  - [在线demo](#在线demo)
  - [项目结构](#项目结构)
  - [当前进展](#当前进展)
  - [模型相关](#模型相关)
    - [各个版本ONNX模型效果对比](#各个版本onnx模型效果对比)
      - [文本检测模型(仅供参考)](#文本检测模型仅供参考)
      - [文本识别模型(仅供参考)](#文本识别模型仅供参考)
      - [模型转onnx](#模型转onnx)
  - [原始发起者及初创作者](#原始发起者及初创作者)
  - [版权声明](#版权声明)
  - [授权](#授权)
  - [联系我们](#联系我们)
  - [示例图](#示例图)
      - [C++/JVM示例图像](#cjvm示例图像)
      - [.Net示例图像](#net示例图像)
      - [多语言示例图像](#多语言示例图像)
</details>


## 简介
- 💖目前已知**运行速度最快、支持最广**，完全开源免费并支持离线部署的多平台多语言OCR SDK
- **中文广告**： 欢迎加入我们的QQ群下载模型及测试程序，qq群号：887298230
- **缘起**：百度paddlepaddle工程化不是太好，为了方便大家在各种端上进行ocr推理，我们将它转换为onnx格式，使用``python/c++/java/swift/c#`` 将它移植到各个平台。

- **名称来源**： 轻快好省并智能。 基于深度学习技术的OCR技术，主打人工智能优势及小模型，以速度为使命，效果为主导。

- 基于百度的开源PaddleOCR 模型及训练，任何人可以使用本推理库，也可以根据自己的需求使用百度的paddlepaddle框架进行模型优化。

## 近期更新([more](./docs/change_log.md))
#### 📌2022-06-25 update:
- 重新整理python部分推理代码，将常用调节参数全部放到yaml文件中，便于调节，更加容易使用,详情参见：[README](./python/README.md)
- 之前旧的推理代码位于分支：[old_python_infer](https://github.com/RapidAI/RapidOCR/tree/old_python_infer)

#### 🥟2022-05-25 update
- 增加基于PaddleOCR v3 rec的python版本推理代码，详情参见[v3 rec](./python/rapidocr_onnxruntime/ch_ppocr_v3_rec)

#### 🍿2022-05-15 update
- 增加PaddleOCR v3 rec模型转换后的ONNX模型，直接去网盘下载替换即可。([百度网盘](https://pan.baidu.com/s/1PTcgXG2zEgQU6A_A3kGJ3Q?pwd=jhai) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing))
- 增加文本识别模型各个版本效果对比表格，详情点击[各个版本ONNX模型效果对比](#各个版本onnx模型效果对比)。v3的文本识别模型从自己构建测试集上的指标来看不如之前的好。

#### 😀2022-05-12 upadte
- 增加PaddleOCR v3 det模型转换的ONNX模型，直接去网盘下载，替换即可。([百度网盘](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing))
- 增加各个版本文本检测模型效果对比表格，详情点击[各个版本ONNX模型效果对比](#各个版本onnx模型效果对比)。v3的文本检测模型从自己构建测试集上的指标来看是好于之前的v2的，推荐使用。

#### 🎧2022-04-04 update
- 增加python下的基于OpenVINO推理引擎的支持
- 给出OpenVINO和ONNXRuntime的性能对比表格，详情参见:[python/rapidocr_openvino/README](./python/rapidocr_openvino/README.md)


## 整个框架
<div align="center">
    <img src="./assets/RapidOCR_system.jpg">
</div>

## 常见问题  [FAQ](./docs/FAQ.md)

## SDK 编译状态
鉴于ubuntu用户都是商业用户，也有编译能力，暂不提供预编译包使用，可自行编译。

| 平台            | 编译状态 |   提供状态 |
| --------------- | -------- | -------- |
| Windows x86/x64 |  [![CMake-windows-x86-x64](https://github.com/RapidAI/RapidOCR/actions/workflows/windows-all-build.yaml/badge.svg)](https://github.com/RapidAI/RapidOCR/actions/workflows/windows-all-build.yaml)        |  [下载链接](https://github.com/RapidAI/RapidOCR/releases) |
| Linux x64       |  [![CMake-linux](https://github.com/RapidAI/RapidOCR/actions/workflows/make-linux.yml/badge.svg)](https://github.com/RapidAI/RapidOCR/actions/workflows/make-linux.yml) |  暂不提供，自行编译 |

## [在线demo](http://rapidocr.51pda.cn:9003/)
- **说明**: 本在线demo不存储小伙伴们上传测试的任何图像数据
- **demo所用模型组合为**: `ch_PP-OCRv3_det + ch_ppocr_mobile_v2.0_cls +  ch_ppocr_mobile_v2.0_rec`
- **运行机器配置**: `4核 AMD EPYC 7K62 48-Core Processor `
- **示例图**:
    <div align="center">
        <img src="./assets/demo.gif" width="100%" height="100%">
    </div>

## 项目结构
<details>
    <summary>(点击展开)</summary>

    RapidOCR
    ├── android             # 安卓工程目录
    ├── api4cpp             # c语言跨平台接口库源码目录，直接用根下的CMakelists.txt 编译
    ├── assets              # 一些演示用的图片，不是测试集
    ├── commonlib           # 通用库
    ├── cpp                 # 基于c++的工程项目文件夹
    ├── dotnet              # .Net程序目录
    ├── FAQ.md              # 一些问答整理
    ├── images              # 测试用图片，典型的测试图，一张是自然场景
    ├── include             # 编译c语言接口库时的头文件目录
    ├── ios                 # 苹果手机平台工程目录
    ├── jvm                 # 基于java的工程目录
    ├── lib                 # 编译用库文件目录，用于编译c语言接口库用，默认并不上传二进制文件
    ├── ocrweb              # 基于python和Flask web
    ├── python              # python推理代码目录
    ├── release             # 发布的sdk
    └── tools               #  一些转换脚本之类

</details>


## 当前进展
- [x] C++范例(Windows/Linux/macOS): [demo](./cpp)
- [x] Jvm范例(Java/Kotlin): [demo](./jvm)
- [x] .Net范例(C#): [demo](./dotnet)
- [x] Android范例: [demo](./android)
- [x] python范例: [demo](./python)
- [x] OpenVINO加速版本，进行中
- [ ] IOS范例: 等待有缘人贡献代码
- [ ] 依据python版本重写C++推理代码，以提升推理效果，并增加对gif/tga/webp 格式图片的支持

## 模型相关
- 可以直接下载使用的模型 ([百度网盘](https://pan.baidu.com/s/1PTcgXG2zEgQU6A_A3kGJ3Q?pwd=jhai) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)）

|模型名称|模型简介|模型大小|备注|
|:---:|:---:|:---:|:---:|
|⭐ ch_PP-OCRv3_det_infer.onnx|轻量文本检测模型|2.23M|较v1轻量检测，精度有较大提升 from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/README_ch.md#pp-ocr%E7%B3%BB%E5%88%97%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8%E6%9B%B4%E6%96%B0%E4%B8%AD)|
|⭐ ch_PP-OCRv2_rec_infer.onnx|轻量文本识别模型|7.79M||

### 各个版本ONNX模型效果对比
#### 文本检测模型(仅供参考)
- 测试集：自己构建`中英文(111个，包含卡证、文档和自然图像)`

|                模型                  | infer_Speed(s/img) | precision | recall | hmean  | 模型大小 |
| :---------------------------------: | :----------------: | :-------: | :----: | :----: | :------: |
| ch_ppocr_mobile_v2.0_det_infer.onnx |     0.4345742      |  0.7277   | 0.8413 | 0.7785 |   2.3M   |
|     ch_PP-OCRv2_det_infer.onnx      |     0.5116553      |  0.7817   | 0.8472 | 0.8123 |   2.3M   |
|     ch_PP-OCRv3_det_infer.onnx      |     0.5723512      |  **0.7740**   | **0.8837** | **0.8237** |   2.4M   |

#### 文本识别模型(仅供参考)
- 测试集: 自己构建`中英文(168个)`

|                模型                 | infer_Speed(s/img)   | Score     |    Exact_Match   |   Char_Match | 模型大小 |
| :---------------------------------: | :------------------: | :-------: | :--------------: | :-------------: | :--: |
| ch_ppocr_mobile_v2.0_rec_infer.onnx |       0.0111        |  **0.7287**   |      **0.5595**      |     0.8979      | 4.3M |
|     ch_PP-OCRv2_rec_infer.onnx      |       0.0193        |  0.6955   |      0.4881      |     **0.9029**      | 8.0M |
|     ch_PP-OCRv3_rec_infer.onnx      |       0.0145        |  0.5537   |      0.3274      |     0.7800      |  11M |
| ch_PP-OCRv3_rec_train_student.onnx  |       0.0157        |  0.5537   |      0.3274      |     0.7800      | 11M  |
| ch_PP-OCRv3_rec_train_teacher.onnx  |       0.0140        |  0.5381   |      0.3095      |     0.7667      | 11M  |


#### 模型转onnx
  - [PaddleOCRModelConverter](https://github.com/RapidAI/PaddleOCRModelConverter) by @[SWHL](https://github.com/SWHL)
  - [Paddle2OnnxConvertor](https://github.com/RapidAI/Paddle2OnnxConvertor) by @[benjaminwan](https://github.com/benjaminwan)
  - [手把手教你使用ONNXRunTime部署PP-OCR](https://aistudio.baidu.com/aistudio/projectdetail/1479970?channelType=0&channel=0) by @[Channingss](https://github.com/Channingss)


## 原始发起者及初创作者
<p align="left">
    <a href="https://github.com/benjaminwan"><img src="https://avatars.githubusercontent.com/u/2362051?v=4" width=65 height=65></a>
    <a href="https://github.com/znsoftm"><img src="https://avatars.githubusercontent.com/u/15354249?v=4" width=65 height=65></a>
    <a href="https://github.com/SWHL"><img src="https://avatars.githubusercontent.com/u/28639377?v=4" width=65 height=65></a>
</p>

## 版权声明
- 如果你的产品使用了本仓库中的全部或部分代码、文字或材料
- 请注明出处并包括我们的github url: `https://github.com/RapidAI/RapidOCR`

## 授权
- OCR模型版权归百度所有，其它工程代码版权归本仓库所有者所有。
- 本软件采用Apache 授权方式，欢迎大家贡献代码，提交issue 甚至pr.

## 联系我们
- 您可以通过QQ群联系到我们：**887298230**
- 群号搜索不到时，请直接点此[**链接**](https://jq.qq.com/?_wv=1027&k=P9b3olx6)，找到组织
- 用QQ扫描以下二维码:

    <div align="center">
        <img src="./assets/qq_team.bmp" width="25%" height="25%" align="center">
    </div>

## 示例图
#### C++/JVM示例图像
<div align="center">
    <img src="./assets/demo_cpp.png" width="100%" height="100%">
</div>

#### .Net示例图像
<div align="center">
    <img src="./assets/demo_cs.png" width="100%" height="100%">
</div>

#### 多语言示例图像
<div align="center">
    <img src="./assets/demo_multi_language.png" width="80%" height="80%">
</div>
