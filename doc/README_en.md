
<div align="center">
  <img src="./assets/RapidOCR_LOGO.png" width="45%" height="45%"/>
</div>

# RapidOCR

[简体中文](README.md) | English

<p align="left">
    <a href="https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/assets/RapidOCRDemo.ipynb" target="_blank"><img src="./assets/colab-badge.svg" alt="Open in Colab"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python-3.6+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors"><img src="https://img.shields.io/github/contributors/RapidAI/RapidOCR?color=9ea"></a>
    <a href="https://github.com/RapidAI/RapidOCR/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidOCR?color=ccf"></a>
</p>

<details>
    <summary>Contents</summary>

- [RapidOCR](#rapidocr)
  - [Introduction](#introduction)
  - [Recently updates(more)](#recently-updatesmore)
      - [🍿2022-05-15 update](#2022-05-15-update)
      - [😀2022-05-12 upadte](#2022-05-12-upadte)
      - [🎧2022-04-04 update](#2022-04-04-update)
  - [FAQ](#faq)
  - [SDK compilation status](#sdk-compilation-status)
  - [Online demo](#online-demo)
  - [Directory structure](#directory-structure)
  - [Current Progress](#current-progress)
  - [Model related](#model-related)
      - [Download models( Google Drive|[Baidu NetDisk](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) )](#download-models-google-drivebaidu-netdisk-)
      - [Model to onnx](#model-to-onnx)
    - [Compared](#compared)
      - [Text Det](#text-det)
      - [Text Recognition](#text-recognition)
  - [PaddleOCR-FAQ](#paddleocr-faq)
  - [Original initiator and start-up author](#original-initiator-and-start-up-author)
  - [Authorization](#authorization)
  - [Contact us](#contact-us)
  - [Demo](#demo)
      - [Demonstration with C++/JVM](#demonstration-with-cjvm)
      - [Demonstration with .Net](#demonstration-with-net)
      - [Demonstratioin with multi_language](#demonstratioin-with-multi_language)
</details>


## Introduction
- Completely open source, free and support offline deployment of multi-platform and multi-language OCR SDK
- **Chinese Advertising**: Welcome to join our QQ group to download the model and test program, QQ group number: 887298230
- **Cause**: Baidu paddlepaddle engineering is not very good, in order to facilitate everyone to perform OCR reasoning on various terminals, we convert it to onnx format, use ``python/c++/java/swift/c#'' to change It is ported to various platforms.

- **Name Source**: Light, fast, economical and smart. OCR technology based on deep learning technology focuses on artificial intelligence advantages and small models, with speed as the mission and effect as the leading role.

- Based on Baidu's open source PaddleOCR model and training, anyone can use this inference library, or use Baidu's paddlepaddle framework for model optimization according to their own needs.

## Recently updates([more](./change_log_en.md))
#### 🍿2022-05-15 update
- Add the ONNX model converted from the PaddleOCR v3 rec model, just go to the network disk to download and replace it. ([Baidu Netdisk](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing ))
- Added a comparison table of the effects of each version of the text recognition model. For details, click [Comparison of the effects of various versions of ONNX models] (#Comparison of the effects of various versions of onnx models). The text recognition model of v3 is not as good as the previous one in terms of the indicators on the test set constructed by itself.

#### 😀2022-05-12 upadte
- Add the ONNX model converted from the PaddleOCR v3 det model, download it directly from the network disk, and replace it. ([Baidu Netdisk](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing ))
- Added a comparison table of text detection model effects of various versions. For details, click [Comparison of the effects of various versions of ONNX models] (#Comparison of the effects of various versions of onnx models). The text detection model of v3 is better than the previous v2 in terms of the indicators on the test set constructed by itself, and it is recommended to use it.

#### 🎧2022-04-04 update
- Add suport for OpenVINO under python
- Give the performance comparison table of OpenVINO and ONNXRuntime
- For details:[python/README](./python/README.md)

## [FAQ](./doc/FAQ.md)

## SDK compilation status
Since ubuntu users are all commercial users and have the ability to compile, pre-compiled packages are not provided for the time being, and they can be compiled by themselves.

| Platform | Compilation Status | Offer Status |
| --------------- | -------- | -------- |
| Windows x86/x64 | [![CMake-windows-x86-x64](https://github.com/RapidOCR/RapidOCR/actions/workflows/windows-all-build.yaml/badge.svg)](https://github.com/RapidOCR/RapidOCR/actions/workflows/windows-all-build.yaml) | [Download Link](https://github.com/RapidAI/RapidOCR/releases) |
| Linux x64 | [![CMake-linux](https://github.com/RapidOCR/RapidOCR/actions/workflows/make-linux.yml/badge.svg)](https://github.com/RapidOCR/RapidOCR/actions/workflows/make-linux.yml) | Not available yet, compile by yourself |

## Online demo
- [Web demo](http://rapidocr.51pda.cn:9003/)
- The model combination used in the demo is: **ch_PP-OCRv2 det** + **mobile cls** + **mobile rec**
- Environment: `4 Core AMD EPYC 7K62 48-Core Processor `
- **Gif Demo**:
    <div align="center">
        <img src="./assets/demo.gif" width="100%" height="100%">
    </div>

## Directory structure
<details>
    <summary>click to expand</summary>

    RapidOCR
        ├── android         # Android project directory
        ├── api4cpp         # C language cross-platform interface library source code directory, directly compile with CMakelists.txt under the root
        ├── assets          # Some pictures for demonstration, not a test set
        ├── commonlib       # common library
        ├── cpp             # C++-based project folder
        ├── datasets        # Additional training datasets
        ├── dotnet          # .Net program directory
        ├── FAQ.md          # Some questions and answers
        ├── images          # Test pictures, two typical test pictures, one is a natural scene, the other is a long text
        ├── include         # The header file directory when compiling the c language interface library
        ├── ios             # Apple mobile phone platform project directory
        ├── jvm             # java-based project directory
        ├── lib             # Compilation library file directory, used to compile the C language interface library. Binary files are not uploaded by default
        ├── ocrweb          # Based on python and Flask web
        ├── python          # python reasoning code directory
        ├── release         #
        └── tools           # Some conversion scripts and the like

</details>


## Current Progress
- [x] C++ example (Windows/Linux/macOS): [demo](./cpp)
- [x] Jvm example (Java/Kotlin): [demo](./jvm)
- [x] .Net example (C#): [demo](./dotnet)
- [x] Android example: [demo](./android)
- [x] python example: [demo](./python)
- [ ] IOS example: waiting for someone to contribute code
- [ ] Rewrite the C++ reasoning code according to the python version to improve the reasoning effect, and add support for gif/tga/webp format pictures

## Model related
#### Download models( [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)|[Baidu NetDisk](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) )

#### Model to onnx
   - ⭐[PaddleOCRModelConverter](https://github.com/RapidAI/PaddleOCRModelConverter) by @[SWHL](https://github.com/SWHL)
   - [Paddle2OnnxConvertor](https://github.com/RapidAI/Paddle2OnnxConvertor) by @[benjaminwan](https://github.com/benjaminwan)
   - [Teach you to use ONNXRunTime to deploy PP-OCR](https://aistudio.baidu.com/aistudio/projectdetail/1479970?channelType=0&channel=0) by @Channingss

### Compared
#### Text Det
- test dataset: `Chinese and English (111, including cards, documents and natural images)`

|                Model                  | infer_Speed(s/img) | precision | recall | hmean  | Model Size |
| :---------------------------------: | :----------------: | :-------: | :----: | :----: | :------: |
| ch_ppocr_mobile_v2.0_det_infer.onnx |     0.4345742      |  0.7277   | 0.8413 | 0.7785 |   2.3M   |
|     ch_PP-OCRv2_det_infer.onnx      |     0.5116553      |  0.7817   | 0.8472 | 0.8123 |   2.3M   |
|     ch_PP-OCRv3_det_infer.onnx      |     0.5723512      |  **0.7740**   | **0.8837** | **0.8237** |   2.4M   |

#### Text Recognition
- test dataset: `Chinese and English (168)`

|                Model                 | infer_speed(s/img)   | Score     |    Exact_Match   |   Char_Match | Model Size |
| :---------------------------------: | ------------------: | :-------: | :--------------: | :-------------: | :--: |
| ch_ppocr_mobile_v2.0_rec_infer.onnx |       0.0111        |  **0.7287**   |      **0.5595**      |     0.8979      | 4.3M |
|     ch_PP-OCRv2_rec_infer.onnx      |       0.0193        |  0.6955   |      0.4881      |     **0.9029**      | 8.0M |
|     ch_PP-OCRv3_rec_infer.onnx      |       0.0145        |  0.5537   |      0.3274      |     0.7800      |  11M |
| ch_PP-OCRv3_rec_train_student.onnx  |       0.0157        |  0.5537   |      0.3274      |     0.7800      | 11M  |
| ch_PP-OCRv3_rec_train_teacher.onnx  |       0.0140        |  0.5381   |      0.3095      |     0.7667      | 11M  |

## [PaddleOCR-FAQ](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/FAQ.md)

## Original initiator and start-up author
<p align="left">
    <a href="https://github.com/benjaminwan"><img src="https://avatars.githubusercontent.com/u/2362051?v=4" width=65 height=65></a>
    <a href="https://github.com/znsoftm"><img src="https://avatars.githubusercontent.com/u/15354249?v=4" width=65 height=65></a>
    <a href="https://github.com/SWHL"><img src="https://avatars.githubusercontent.com/u/28639377?v=4" width=65 height=65></a>
</p>

## Authorization
- The copyright of the OCR model belongs to Baidu, and the copyright of other engineering codes belongs to the owner of this warehouse.
- This software is licensed under LGPL. You are welcome to contribute code, submit an issue or even pr.

## Contact us
- You can contact us through QQ group: **887298230**
- If you can’t find the group number, please click here [**link**](https://jq.qq.com/?_wv=1027&k=P9b3olx6) to find the organization
- Scan the following QR code with QQ:

    <div align="center">
        <img src="./assets/qq_team.bmp" width="25%" height="25%" align="center">
     </div>

## Demo
#### Demonstration with C++/JVM
<div align="center">
    <img src="./assets/demo_cpp.png" width="100%" height="100%">
</div>

#### Demonstration with .Net
<div align="center">
    <img src="./assets/demo_cs.png" width="100%" height="100%">
</div>

#### Demonstratioin with multi_language
<div align="center">
    <img src="./assets/demo_multi_language.png" width="80%" height="80%">
</div>
