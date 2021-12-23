
<div align="center">
  <img src="./assets/RapidOCR_LOGO.png" width="45%" height="45%"/>
</div>

# RapidOCR

[简体中文](README.md) | English

<p align="left">
    <a href="https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/RapidOCRDemo.ipynb" target="_blank"><img src="./assets/colab-badge.svg" alt="Open in Colab"></a>
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
  - [Recently updates、](#recently-updates)
      - [🎄2021-12-18 update](#2021-12-18-update)
      - [2021-11-28 update](#2021-11-28-update)
      - [2021-11-13 update](#2021-11-13-update)
      - [2021-10-27 update](#2021-10-27-update)
      - [2021-09-13 update](#2021-09-13-update)
      - [2021-09-11 update](#2021-09-11-update)
      - [2021-08-07 update](#2021-08-07-update)
      - [2021-07-17 update](#2021-07-17-update)
      - [2021-07-04 update](#2021-07-04-update)
      - [2021-06-20 update](#2021-06-20-update)
      - [2021-06-10 update](#2021-06-10-update)
      - [2021-06-08 update](#2021-06-08-update)
      - [2021-03-24 update](#2021-03-24-update)
  - [FAQ](#faq)
  - [SDK compilation status](#sdk-compilation-status)
  - [Online demo](#online-demo)
  - [Directory structure](#directory-structure)
  - [Current Progress](#current-progress)
  - [Model related](#model-related)
      - [Download models](#download-models)
      - [Model to onnx](#model-to-onnx)
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

## Recently updates、
#### 🎄2021-12-18 update
- Add [Google Colab Demo](https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/RapidOCRDemo.ipynb).
- Change the default det model of the `python/rapidOCR.sh`

#### 2021-11-28 update
- Update the [ocrweb](http://rapidocr.51pda.cn:9003/) part
  - Add the display of the inference time of each stage.
  - Add docs of the ocrweb.
  - Change the det model(`ch_PP-OCRv2_det_infer.onnx`), faster and more accurate.

#### 2021-11-13 update
- Add adjustable super parameters for text detection and recognition in Python version, mainly `box_thresh|unclip_ratio|text_score`, see [parameter adjustment](python/README.md#相关调节参数) for details
- The dictionary position in text recognition is given in parameter mode to facilitate flexible configuration. See [keys_path](python/rapidOCR.sh) for details

#### 2021-10-27 update
- Add the code that uses the onnxruntime GPU version of infering follow the [official tutorial](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) Configuration. (however, the onnxruntime GPU version is not stable to use)

- See: `python/README.md` for specific steps.

#### 2021-09-13 update
- Add a whl file based on `Python` for ease of use. See `release/python` for details.

#### 2021-09-11 update
- Add `PP-OCRv2` new model onnx version.
- The infering code of the method is unchanged, and the corresponding model can be directly replaced.
- After evaluation on its own test set:
    - The effect of `PP-OCRv2` detection model has been greatly improved, and the model size has not changed.
    - The effect of `PP-OCRv2` recognition model was not significantly improved, and the model size increased by 3.58M.

- Upload the model to [Baidu online disk extraction code: 30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg) or [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)

<details>
    <summary>Previous update logs</summary>

#### 2021-08-07 update
- TODO:
    - [ ] PP structure table structure and cell coordinate prediction are being sorted out.

- Previously done, unfinished, welcome to PR
    - [ ] make dokcer image
    - [x] try onnxruntime GPU reasoning

#### 2021-07-17 update
- Improve the README document
- Add **English, number recognition**onnx model, please refer to `python/en_number_ppocr_mobile_v2_rec` for details, the usage is the same as others
- Organize [Model to onnx](#model-related)

#### 2021-07-04 update
- The python program under the repository can be successfully run on the Raspberry Pi 4B. For more information, please enter the QQ group and ask the group owner
- Update the overall structure diagram and add support for Raspberry Pi

#### 2021-06-20 update
- Optimize the display of recognition results in ocrweb, and add recognition animations to demonstrate at the same time
- Update the `datasets` directory, add some commonly used database links

#### 2021-06-10 update
- Add server version text recognition model, see details [Extract code：30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)

#### 2021-06-08 update
- Organize the warehouse and unify the model download path
- Improve related documentation

#### 2021-03-24 update
- The new model is fully compatible with ONNXRuntime 1.7 or higher. Special thanks: @Channingss
- The performance of the new version of onnxruntime is improved by more than 40% compared to 1.6.0.

</details>


## [FAQ](FAQ.md)

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
#### Download models
   - [Google Drive link](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)
   - (download link: [Baidu extract code: 30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg))
     ```text
     ch_ppocr_mobile_v2.0_det_infer.onnx
     ch_ppocr_mobile_v2.0_cls_infer.onnx
     ch_ppocr_mobile_v2.0_rec_infer.onnx

     ch_ppocr_server_v2.0_det_infer.onnx
     ch_ppocr_server_v2.0_rec_infer.onnx

     japan_rec_crnn.onnx
     en_number_mobile_v2.0_rec_infer.onnx
     ```
#### Model to onnx
   - [Teach you to use ONNXRunTime to deploy PP-OCR](https://aistudio.baidu.com/aistudio/projectdetail/1479970?channelType=0&channel=0) by @Channingss
   - [✧✧PaddleOCRModelConverter](https://github.com/RapidAI/PaddleOCRModelConverter) by @[SWHL](https://github.com/SWHL)
   - [Paddle2OnnxConvertor](https://github.com/RapidAI/Paddle2OnnxConvertor) by @[benjaminwan](https://github.com/benjaminwan)

## [PaddleOCR-FAQ](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_ch/FAQ.md)

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
