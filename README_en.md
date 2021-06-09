
<div align="center">
  <img src="./assets/RapidOCR_LOGO.png" width="45%" height="45%"/>
</div>

# RapidOCR

English | [Chinese](README.md)
<details open>
    <summary>Contents</summary>

- [RapidOCR](#rapidocr)
  - [Introduction](#introduction)
  - [Recent updates](#recent-updates)
      - [2021-06-08 update](#2021-06-08-update)
      - [2021-03-24 update](#2021-03-24-update)
  - [FAQ](#faq)
  - [SDK compilation status](#sdk-compilation-status)
  - [Online demo](#online-demo)
  - [Directory structure](#directory-structure)
  - [Current Progress](#current-progress)
  - [Model Conversion](#model-conversion)
    - [Model Conversion Instructions](#model-conversion-instructions)
    - [PaddleOCR-FAQ](#paddleocr-faq)
    - [ONNX model download](#onnx-model-download)
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

## Recent updates
#### 2021-06-08 update
- Organize the warehouse and unify the model download path
- Improve related documentation

#### 2021-03-24 update
- The new model is fully compatible with ONNXRuntime 1.7 or higher. Special thanks: @Channingss
- The performance of the new version of onnxruntime is improved by more than 40% compared to 1.6.0.

## [FAQ](FAQ.md)

## SDK compilation status
Since ubuntu users are all commercial users and have the ability to compile, pre-compiled packages are not provided for the time being, and they can be compiled by themselves.

| Platform | Compilation Status | Offer Status |
| --------------- | -------- | -------- |
| Windows x86/x64 | [![CMake-windows-x86-x64](https://github.com/RapidOCR/RapidOCR/actions/workflows/windows-all-build.yaml/badge.svg)](https://github.com/RapidOCR/RapidOCR/actions/workflows/windows-all-build.yaml) | Download on the right |
| Linux x64 | [![CMake-linux](https://github.com/RapidOCR/RapidOCR/actions/workflows/make-linux.yml/badge.svg)](https://github.com/RapidOCR/RapidOCR/actions/workflows/make-linux.yml) | Not available yet, compile by yourself |

## Online demo
- [Web demo](http://rapidocr.51pda.cn:9003/)
- The model combination used in the demo is: **server det** + **mobile cls** + **mobile rec**

## Directory structure
<details>
    <summary>click to expand</summary>

   RapidOCR
    ├── android         # Android project directory
    ├── api4cpp         # C language cross-platform interface library source code directory, directly compile with CMakelists.txt under the root
    ├── assets          # Some pictures for demonstration, not a test set
    ├── commonlib       # common library
    ├── cpp             # C++-based project folder
    ├── datasets        # Additional training datasets, Baidu SkyDrive download address
    ├── dotnet          # .Net program directory
    ├── FAQ.md          # Some questions and answers
    ├── images          # Test pictures, two typical test pictures, one is a natural scene, the other is a long text
    ├── include         # The header file directory when compiling the c language interface library
    ├── ios             # Apple mobile phone platform project directory
    ├── jvm             # java-based project directory
    ├── lib             # Compilation library file directory, used to compile the C language interface library. Binary files are not uploaded by default
    ├── models          # Place available model file download information, based on Baidu network disk
    ├── ocrweb          # Based on python and Flask web
    ├── python          # python reasoning code directory
    ├── release         #
    ├── tools           # Some conversion scripts and the like
    └── training        # Script or program used to train your own model
</details>


## Current Progress
- [x] C++ example (Windows/Linux/macOS): [demo](./cpp)
- [x] Jvm example (Java/Kotlin): [demo](./jvm)
- [x] .Net example (C#): [demo](./dotnet)
- [x] Android example: [demo](./android)
- [x] python example: [demo](./python)
- [ ] IOS example: waiting for someone to contribute code
- [ ] Rewrite the C++ reasoning code according to the python version to improve the reasoning effect, and add support for gif/tga/webp format pictures

## Model Conversion
- The models currently supported by the conversion script:
     - 1 text direction classification model,
     - 2 detection models,
     - 28 recognition models (2 in simplified Chinese, 26 in traditional Chinese, etc.), totaling 31
### [Model Conversion Instructions](./models)

### [PaddleOCR-FAQ](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/FAQ.md)

### ONNX model download
- [Extraction code: 30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)

## Original initiator and start-up author
- [benjaminwan](https://github.com/benjaminwan)
- [znsoftm](https://github.com/znsoftm)

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
