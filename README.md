
<div align="center">
  <img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/RapidOCR_LOGO.png" width="65%" height="65%"/>
</div>

# RapidOCR (捷智OCR- 信创级开源OCR - 为世界内容安全贡献力量)

简体中文 | [English](https://github.com/RapidAI/RapidOCR/blob/main/docs/README_en.md)

<p align="left">
    <a href="https://rapidai.deepdatasec.com:9003/" target="_blank"><img src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E-Online%20Demo-blue"></a>
    <a href="https://huggingface.co/spaces/SWHL/RapidOCRDemo" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
    <a href="https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/assets/RapidOCRDemo.ipynb" target="_blank"><img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/colab-badge.svg" alt="Open in Colab"></a>
    <a href="https://aistudio.baidu.com/aistudio/projectdetail/4444785?sUid=57084&shared=1&ts=1660896122332" target="_blank"><img src="https://img.shields.io/badge/PP-Open in AI Studio-blue.svg"></a><br/>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors"><img src="https://img.shields.io/github/contributors/RapidAI/RapidOCR?color=9ea"></a>
    <a href="https://pepy.tech/project/rapidocr_onnxruntime"><img src="https://static.pepy.tech/personalized-badge/rapidocr_onnxruntime?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Ort"></a>
    <a href="https://pypi.org/project/rapidocr-onnxruntime/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr-onnxruntime"></a>
    <a href="https://github.com/RapidAI/RapidOCR/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidOCR?color=ccf"></a>
    <a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
</p>


<details>
    <summary>目录</summary>

- [RapidOCR (捷智OCR- 信创级开源OCR - 为世界内容安全贡献力量)](#rapidocr-捷智ocr--信创级开源ocr---为世界内容安全贡献力量)
  - [商业支持](#商业支持)
  - [简介](#简介)
  - [文档导航](#文档导航)
  - [近期更新(more)](#近期更新more)
      - [🎮2023-03-11 v1.2.2 update:](#2023-03-11-v122-update)
      - [🧢2023-03-07 v1.2.1 update:](#2023-03-07-v121-update)
      - [⛸2023-02-16 update:](#2023-02-16-update)
  - [生态框架](#生态框架)
  - [在线demo](#在线demo)
  - [TODO](#todo)
  - [原始发起者及初创作者](#原始发起者及初创作者)
  - [致谢](#致谢)
  - [赞助](#赞助)
  - [版权声明](#版权声明)
  - [授权](#授权)
  - [加入我们](#加入我们)
  - [示例图](#示例图)
      - [C++/JVM示例图像](#cjvm示例图像)
      - [.Net示例图像](#net示例图像)
      - [多语言示例图像](#多语言示例图像)
</details>

## 商业支持
- 提供信创平台多架构，包括**Arm/X86/mips(龙芯)/RISC-V**等信创CPU支持，同时兼容**ONNXRuntime/OpenVINO/NCNN**。
- 有意者邮件联系: znsoft@163.com, 请先邮件咨询服务项目，即时回复联系方式。

## 简介
- 💖目前已知**运行速度最快、支持最广**，完全开源免费并支持离线快速部署的多平台多语言OCR。
- **中文广告**： 欢迎加入我们的QQ群下载模型及测试程序，QQ群号：887298230(已满)，2群(755960114)
- **缘起**：百度paddlepaddle工程化不是太好，为了方便大家在各种端上进行ocr推理，我们将它转换为onnx格式，使用`Python/C++/Java/Swift/C#` 将它移植到各个平台。
- **名称来源**： 轻快好省并智能。基于深度学习技术的OCR技术，主打人工智能优势及小模型，以速度为使命，效果为主导。
- **使用**：
  - 如果仓库下已有模型满足要求 → RapidOCR部署使用即可。
  - 不满足要求 → 基于[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)在自己数据上微调 → RapidOCR部署。
- 如果该仓库有帮助到你，还请点个小星星⭐呗！

## 文档导航
- [Python范例](https://github.com/RapidAI/RapidOCR/blob/main/python/README.md)
  - [rapidocr_openvino](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr_openvino/README.md)
- [C++范例(Windows/Linux/macOS)](https://github.com/RapidAI/RapidOCR/blob/main/cpp)
  - [RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx)
  - [RapidOcrNcnn](https://github.com/RapidAI/RapidOcrNcnn)
- [Jvm范例(Java/Kotlin)](https://github.com/RapidAI/RapidOCR/blob/main/jvm)
  - [RapidOcrOnnxJvm](https://github.com/RapidAI/RapidOcrOnnxJvm)
  - [RapidOcrNcnnJvm](https://github.com/RapidAI/RapidOcrNcnnJvm)
- [.Net范例(C#)](https://github.com/RapidAI/RapidOCR/blob/main/dotnet/RapidOcrOnnxCs/README.md)
- [Android范例](https://github.com/RapidAI/RapidOcrAndroidOnnx)
- 网页版范例
  - [网页版OCR](https://github.com/RapidAI/RapidOCR/blob/main/ocrweb/README.md)
  - [Nuitka打包rapdocr_web](https://github.com/RapidAI/RapidOCR/wiki/Nuitka%E6%89%93%E5%8C%85rapidocr_web%E6%93%8D%E4%BD%9C%E6%96%87%E6%A1%A3)
  - [多语言网页版OCR](https://github.com/RapidAI/RapidOCR/blob/main/ocrweb_multi/README.md)
- [版面结构化](https://github.com/RapidAI/RapidOCR/blob/main/python/rapid_structure/README.md)
  - [含文本的图像方向分类](https://github.com/RapidAI/RapidOCR/blob/main/python/rapid_structure/docs/README_Orientation.md)
  - [版面分析](https://github.com/RapidAI/RapidOCR/blob/main/python/rapid_structure/docs/README_Layout.md)
  - [表格还原](https://github.com/RapidAI/RapidOCR/blob/main/python/rapid_structure/docs/README_Table.md)
- 衍生项目
  - [RapidOCR HTTP服务/win32程序/易语言编写](https://github.com/Physton/RapidOCRServer)
- [垂直项目](https://github.com/RapidAI/RapidOCR/blob/main/docs/related_projects.md)
  - [RapidVideOCR](https://github.com/SWHL/RapidVideOCR)：基于RapidOCR，提取视频中的硬字幕
  - [LGPMA_Infer](https://github.com/SWHL/LGPMA_Infer): 表格结构还原 | [博客解读论文和源码](http://t.csdn.cn/QNN3S)
  - [文档图像矫正-PaperEdge](https://github.com/cvlab-stonybrook/PaperEdge) | [Demo](https://huggingface.co/spaces/SWHL/PaperEdgeDemo)
  - [图像文字擦除-CTRNet](https://github.com/lcy0604/CTRNet) | [Demo](https://huggingface.co/spaces/SWHL/CTRNetDemo)
- [模型相关](https://github.com/RapidAI/RapidOCR/blob/main/docs/models.md)
  - [模型转换](https://github.com/RapidAI/RapidOCR/blob/main/docs/models.md#模型转换)
  - [模型下载及效果对比](https://github.com/RapidAI/RapidOCR/blob/main/docs/models.md#模型下载)
- [常见问题 FAQ](https://github.com/RapidAI/RapidOCR/blob/main/docs/FAQ.md)


## 近期更新([more](https://github.com/RapidAI/RapidOCR/blob/main/docs/change_log.md))
#### 🎮2023-03-11 v1.2.2 update:
- 修复实例化python中RapidOCR类传入参数错误

#### 🧢2023-03-07 v1.2.1 update:
- 优化python下rapidocr系列包的接口传入参数，支持实例化类时，动态给定各个参数，更加灵活。
- 如果不指定，则用`config.yaml`下的默认参数。
- 具体可参见：[传入参数](https://github.com/RapidAI/RapidOCR/blob/0a603b4e8919386f3647eca5cdeba7620b4988e0/python/README.md#%E6%8E%A8%E8%8D%90pip%E5%AE%89%E8%A3%85%E5%BF%AB%E9%80%9F%E4%BD%BF%E7%94%A8)

#### ⛸2023-02-16 update:
- 优化ocrweb部分代码，可直接pip安装，快速使用，详情参见[README](https://github.com/RapidAI/RapidOCR/blob/main/ocrweb/README.md)。
- 优化python中各个部分的推理代码，更加紧凑，同时易于维护。


## 生态框架
```mermaid
flowchart LR
    subgraph Step
    direction TB
    C(Text Det) --> D(Text Cls) --> E(Text Rec)
    end

    A[/OurSelf Dataset/] --> B(PaddleOCR) --Train--> Step --> F(PaddleOCRModelConverter)
    F --ONNX--> G{RapidOCR Deploy\n<b>Python/C++/Java/C#</b>}
    G --> H(Windows x86/x64) & I(Linux) & J(Android) & K(Web) & L(Raspberry Pi)

    click B "https://github.com/PaddlePaddle/PaddleOCR" _blank
    click F "https://github.com/RapidAI/PaddleOCRModelConverter" _blank
```

## 在线demo
- [自建在线demo](https://rapidai.deepdatasec.com:9003/)
    - **说明**: 本在线demo不存储小伙伴们上传测试的任何图像数据，详情参见：[ocrweb/README](https://github.com/RapidAI/RapidOCR/blob/main/ocrweb/README.md)
    - **demo所用模型组合（最优组合）为**:
      ```text
      ch_PP-OCRv3_det + ch_ppocr_mobile_v2.0_cls + ch_PP-OCRv3_rec
      ```
    - **运行机器配置**: `4核 AMD EPYC 7K62 48-Core Processor `
    - **示例图**:
        <div align="center">
            <img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/demo.gif" width="100%" height="100%">
        </div>
- [Hugging Face Demo](https://huggingface.co/spaces/SWHL/RapidOCRDemo)
  - 该demo依托于Hugging Face的Spaces构建，采用Gradio库生成，同时添加三个超参数:
    - `box_thresh`: 检测到的框是文本的概率，值越大，框中是文本的概率就越大
    - `unclip_ratio`: 控制检测到文本框的大小，值越大，检测框整体越大
    - `text_score`: 文本识别结果是正确的置信度，值越大，显示出的识别结果更准确
  - 示例图：
    <div align="center">
        <img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/huggingfacedemo.jpg" width="100%" height="100%">
    </div>

## TODO
- [ ] iOS范例: 等待有缘人贡献代码
- [ ] 依据python版本重写C++推理代码，以提升推理效果，并增加对gif/tga/webp 格式图片的支持


## 原始发起者及初创作者
<p align="left">
  <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=RapidAI/RapidOCR" width="50%"/>
  </a>
</p>


## 致谢
- 非常感谢[DeliciaLaniD](https://github.com/DeliciaLaniD)修复ocrweb中扫描动画起始位置错位问题。
- 非常感谢[zhsunlight](https://github.com/zhsunlight)关于参数化调用GPU推理的建议以及细致周到的测试。
- 非常感谢[lzh111222334](https://github.com/lzh111222334)修复python版本下rec前处理部分bug。
- 非常感谢[AutumnSun1996](https://github.com/AutumnSun1996)在[#42](https://github.com/RapidAI/RapidOCR/issues/42)中的建议。
- 非常感谢[DeadWood8](https://github.com/DeadWood8)提供了[Nuitka打包rapidocr_web的操作文档和可执行exe](https://github.com/RapidAI/RapidOCR/wiki/Nuitka%E6%89%93%E5%8C%85rapidocr_web%E6%93%8D%E4%BD%9C%E6%96%87%E6%A1%A3)。

## 赞助
|赞助者|应用的产品|
|:---:|:---:|
|<a href="https://github.com/cuiliang" title="cuiliang"><img src="https://avatars.githubusercontent.com/u/1972649?v=4" width=65 height=65></a>|<a href="https://getquicker.net/" title="Quicker指尖工具箱"><img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Quicker.jpg" width=65 height=65></a>|
|<a href="https://github.com/Eunsolfs" title="Eunsolfs"><img src="https://avatars.githubusercontent.com/u/53815751?v=4" width=65 height=65></a>| - |

- 如果您想要赞助该项目，可直接点击当前页最上面的Sponsor按钮，请写好备注(**您的Github账号名称**)，方便添加到上面赞助列表中。


## 版权声明
- 如果你的产品使用了本仓库中的全部或部分代码、文字或材料
- 请注明出处并包括我们的github url: `https://github.com/RapidAI/RapidOCR`

## 授权
- OCR模型版权归百度所有，其它工程代码版权归本仓库所有者所有。
- 本软件采用Apache 授权方式，欢迎大家贡献代码，提交issue 甚至PR。

## 加入我们
- 微信扫描以下二维码，关注**RapidAI公众号**，回复OCR即可加入RapidOCR微信交流群：
    <div align="center">
        <img src="https://raw.githubusercontent.com/RapidAI/.github/main/assets/RapidAI_WeChatAccount.jpg" width="25%" height="25%" align="center">
    </div>

- 可以通过QQ群加入我们：**755960114**，或者用QQ扫描以下二维码:

    <div align="center">
        <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/qq_group2.png" width="25%" height="25%" align="center">
    </div>

## 示例图
#### C++/JVM示例图像
<div align="center">
    <img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/demo_cpp.png" width="100%" height="100%">
</div>

#### .Net示例图像
<div align="center">
    <img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/demo_cs.png" width="100%" height="100%">
</div>

#### 多语言示例图像
<div align="center">
    <img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/demo_multi_language.jpg" width="80%" height="80%">
</div>
