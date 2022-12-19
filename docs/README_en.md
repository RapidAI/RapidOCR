
<div align="center">
  <img src="../assets/RapidOCR_LOGO.png" width="55%" height="55%"/>
</div>

# RapidOCR （Open source OCR for the security of the digital world)

[简体中文](../README.md) | English

<p align="left">
    <a href="https://rapidai.deepdatasec.com:9003/"><img src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E-Online%20Demo-blue"></a>
    <a href="https://huggingface.co/spaces/SWHL/RapidOCRDemo"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
    <a href="https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/assets/RapidOCRDemo.ipynb" target="_blank"><img src="../assets/colab-badge.svg" alt="Open in Colab"></a>
    <a href="https://aistudio.baidu.com/aistudio/projectdetail/4444785?sUid=57084&shared=1&ts=1660896122332"><img src="https://img.shields.io/badge/PP-Open in AI Studio-blue.svg"></a><br/>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors"><img src="https://img.shields.io/github/contributors/RapidAI/RapidOCR?color=9ea"></a>
    <a href="https://pypi.org/project/rapidocr-onnxruntime/"><img src="https://img.shields.io/pypi/dm/rapidocr-onnxruntime?color=9cf"></a>
    <a href="https://pypi.org/project/rapidocr-onnxruntime/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr-onnxruntime?style=plastic"></a>
    <a href="https://github.com/RapidAI/RapidOCR/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidOCR?color=ccf"></a>
</p>

<details>
    <summary>Contents</summary>

- [RapidOCR （Open source OCR for the security of the digital world)](#rapidocr-open-source-ocr-for-the-security-of-the-digital-world)
  - [Introduction](#introduction)
  - [Navigation](#navigation)
  - [Recently updates(more)](#recently-updatesmore)
      - [⚽2022-12-19 update:](#2022-12-19-update)
      - [🤖2022-12-14 update:](#2022-12-14-update)
      - [🧻2022-11-20 update:](#2022-11-20-update)
  - [Overall Framework](#overall-framework)
  - [SDK compilation status](#sdk-compilation-status)
  - [Demo](#demo)
  - [TODO](#todo)
  - [Model related](#model-related)
      - [Download models(Baidu NetDisk | Google Drive)](#download-modelsbaidu-netdisk--google-drive)
      - [Model to onnx](#model-to-onnx)
    - [Compared](#compared)
      - [Text Det](#text-det)
      - [Text Recognition](#text-recognition)
  - [Original initiator and start-up author](#original-initiator-and-start-up-author)
  - [Acknowledgements](#acknowledgements)
  - [Sponsor](#sponsor)
  - [Authorization](#authorization)
  - [Contact us](#contact-us)
  - [Demo](#demo-1)
      - [Demonstration with C++/JVM](#demonstration-with-cjvm)
      - [Demonstration with .Net](#demonstration-with-net)
      - [Demonstratioin with multi\_language](#demonstratioin-with-multi_language)
</details>


## Introduction
- Completely open source, free and support offline deployment of multi-platform and multi-language OCR.
- **Chinese Advertising**: Welcome to join our QQ group to download the model and test program, QQ group number: 887298230
- **Cause**: Baidu paddlepaddle engineering is not very good, in order to facilitate everyone to perform OCR reasoning on various terminals, we convert it to onnx format, use `Python/C++/Java/Swift/C#` to change It is ported to various platforms.

- **Name Source**: Light, fast, economical and smart. OCR technology based on deep learning technology focuses on artificial intelligence advantages and small models, with speed as the mission and effect as the leading role.

- Based on Baidu's open source PaddleOCR model and training, anyone can use this inference library, or use Baidu's PaddlePaddle framework for model optimization according to their own needs.

## Navigation
- [Python demo](../python/README.md)
- [C++ demo(Windows/Linux/macOS)](../cpp)
- [Jvm demo(Java/Kotlin)](../jvm)
- [.Net demo(C#)](../dotnet)
- [Android demo](../android)
- Web demo:
  - [Web OCR](../ocrweb/README.md)
  - [Multi Web OCR](../ocrweb_multi/README.md)
- [Structure](../python/rapid_structure/README.md)
  - [layout](../python/rapid_structure/docs/README_Layout.md)
  - [table recovery](../python/rapid_structure/docs/README_Table.md)
- Derivatives
  - [RapidOCR HTTP service/win32 program/easy language writing](https://github.com/Physton/RapidOCRServer)
- [Related projects](../docs/related_projects.md)
  - [RapidVideOCR](https://github.com/SWHL/RapidVideOCR): Extract hard subtitles in videos based on RapidOCR
  - [LGPMA_Infer](https://github.com/SWHL/LGPMA_Infer): table structure restoration | [blog interpretation papers and source code](http://t.csdn.cn/QNN3S)
- [FAQ](../docs/FAQ.md)

## Recently updates([more](./change_log_en.md))
#### ⚽2022-12-19 update:
- \[python\] Add the table recovery module, See [Rapid Table](../python/rapid_structure/docs/README_Table.md) for details.

#### 🤖2022-12-14 update:
- \[python\] Move the configuration parameters and model into the module, and at the same time put the model into the whl package, which can be directly installed and used by pip, which is more convenient and quicker.
- For details, see: [README](../python/README.md#推荐pip安装快速使用)

#### 🧻2022-11-20 update:
- \[python\] Add the layout analysis part, which supports the detection and analysis of three layouts: Chinese, English and tables. See: [Rapid Structure](../python/rapid_structure/README.md) section for details.


## Overall Framework
```mermaid
flowchart LR
    subgraph Step
    direction TB
    C(Text Det) --> D(Text Cls) --> E(Text Rec)
    end

    A[/OurSelf Dataset/] --> B(PaddleOCR) --Train--> Step --Convert--> F(ONNX)
    F --> G{RapidOCR Deploy\n<b>Python/C++/Java/C#</b>}
    G --> H(Windows x86/x64) & I(Linux) & J(Android) & K(Web) & L(Raspberry Pi)

    click B "https://github.com/PaddlePaddle/PaddleOCR" _blank
```

## SDK compilation status
Since ubuntu users are all commercial users and have the ability to compile, pre-compiled packages are not provided for the time being, and they can be compiled by themselves.

| Platform | Compilation Status | Offer Status |
| --------------- | -------- | -------- |
| Windows x86/x64 | [![CMake-windows-x86-x64](https://github.com/RapidAI/RapidOCR/actions/workflows/windows-all-build.yaml/badge.svg)](https://github.com/RapidAI/RapidOCR/actions/workflows/windows-all-build.yaml) | [Download Link](https://github.com/RapidAI/RapidOCR/releases) |
| Linux x64 | [![CMake-linux](https://github.com/RapidAI/RapidOCR/actions/workflows/make-linux.yml/badge.svg)](https://github.com/RapidAI/RapidOCR/actions/workflows/make-linux.yml) | Not available yet, compile by yourself |

## Demo
- [Online demo](https://rapidai.deepdatasec.com:9003/)
    - If the demo fails, you can visit the demo on Hugging Face: [RapidOCRDemo](https://huggingface.co/spaces/SWHL/RapidOCRDemo)
    - **Note**: This online demo does not store any image data uploaded and tested by friends. For details, please refer to: [ocrweb/README](../ocrweb/README.md)
    - The model combination (optimal combination) used for the demo is: `ch_PP-OCRv3_det + ch_ppocr_mobile_v2.0_cls + ch_PP-OCRv3_rec`
    - Environment: `4 Core AMD EPYC 7K62 48-Core Processor `
    - **Gif Demo**:
        <div align="center">
            <img src="../assets/demo.gif" width="100%" height="100%">
        </div>
- [Hugging Face Demo](https://huggingface.co/spaces/SWHL/RapidOCRDemo)
  - The demo is built on Hugging Face's Spaces, generated by the Gradio library, and adds three hyperparameters:
       - `box_thresh`: The probability that the detected box is text, the larger the value, the higher the probability that the box is text.
       - `unclip_ratio`: Controls the size of the detected text box, the larger the value, the larger the detection box as a whole.
       - `text_score`: The confidence that the text recognition result is correct, the larger the value, the more accurate the displayed recognition result.
  - Demo:
    <div align="center">
        <img src="../assets/huggingfacedemo.jpg" width="100%" height="100%">
    </div>


## TODO
- [ ] iOS example: waiting for someone to contribute code
- [ ] Rewrite the C++ reasoning code according to the python version to improve the reasoning effect, and add support for gif/tga/webp format pictures

## Model related
#### Download models([Baidu NetDisk](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing))

#### Model to onnx
   - [PaddleOCRModelConverter](https://github.com/RapidAI/PaddleOCRModelConverter) by @[SWHL](https://github.com/SWHL)
   - [Paddle2OnnxConvertor](https://github.com/RapidAI/Paddle2OnnxConvertor) by @[benjaminwan](https://github.com/benjaminwan)
   - [Teach you to use ONNXRunTime to deploy PP-OCR](https://aistudio.baidu.com/aistudio/projectdetail/1479970?channelType=0&channel=0) by @[Channingss](https://github.com/Channingss)

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

## Original initiator and start-up author
<p align="left">
    <a href="https://github.com/benjaminwan"><img src="https://avatars.githubusercontent.com/u/2362051?v=4" width=65 height=65></a>
    <a href="https://github.com/znsoftm"><img src="https://avatars.githubusercontent.com/u/15354249?v=4" width=65 height=65></a>
    <a href="https://github.com/SWHL"><img src="https://avatars.githubusercontent.com/u/28639377?v=4" width=65 height=65></a>
</p>

## Acknowledgements
- Many thanks to [DeliciaLaniD](https://github.com/DeliciaLaniD) for fixing the misplaced start position of scan animation in ocrweb.
- Many thanks to [zhsunlight](https://github.com/zhsunlight) for the suggestion about parameterized call GPU reasoning and the careful and thoughtful testing.
- Many thanks to [lzh111222334](https://github.com/lzh111222334) for fixing some bugs of rec preprocessing under python version.
- Many thanks to [AutumnSun1996](https://github.com/AutumnSun1996) for the suggestiong in the [#42](https://github.com/RapidAI/RapidOCR/issues/42).

## Sponsor

|Sponsor|Applied Products|
|:---:|:---:|
|<a href="https://github.com/cuiliang" title="cuiliang"><img src="https://avatars.githubusercontent.com/u/1972649?v=4" width=65 height=65></a>|<a href="https://getquicker.net/" title="Quicker"><img src="https://tvax2.sinaimg.cn/crop.0.0.600.600.180/82cedfe9ly8h0gd3koj1qj20go0goq34.jpg?KID=imgbed,tva&Expires=1657299650&ssig=7MKPeFM6RU" width=65 height=65></a>|
|<a href="https://github.com/Eunsolfs" title="Eunsolfs"><img src="https://avatars.githubusercontent.com/u/53815751?v=4" width=65 height=65></a>| - |

- If you want to sponsor the project, you can directly click the Sponsor button at the top of the current page, please write a note (e.g. your github account name) to facilitate adding to the sponsorship list above.

## Authorization
- The copyright of the OCR model belongs to Baidu, and the copyright of other engineering codes belongs to the owner of this warehouse.
- This software is licensed under LGPL. You are welcome to contribute code, submit an issue or even pr.

## Contact us
- You can contact us through QQ group: **887298230**
- If you can’t find the group number, please click here [**link**](https://jq.qq.com/?_wv=1027&k=P9b3olx6) to find the organization
- Scan the following QR code with QQ:

    <div align="center">
        <img src="../assets/qq_team.jpg" width="25%" height="25%" align="center">
     </div>

## Demo
#### Demonstration with C++/JVM
<div align="center">
    <img src="../assets/demo_cpp.png" width="100%" height="100%">
</div>

#### Demonstration with .Net
<div align="center">
    <img src="../assets/demo_cs.png" width="100%" height="100%">
</div>

#### Demonstratioin with multi_language
<div align="center">
    <img src="../assets/demo_multi_language.jpg" width="80%" height="80%">
</div>
