<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/RapidOCR_LOGO_white.png"  width="55%" height="55%">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/RapidOCR_LOGO.png"  width="55%" height="55%">
    <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/RapidOCR_LOGO.png">
  </picture>

<div>&nbsp;</div>
<div align="center">
    <b><font size="4"><i>信创级开源OCR - 为世界内容安全贡献力量</i></font></b>
</div>
<div>&nbsp;</div>

<a href="https://huggingface.co/spaces/SWHL/RapidOCRDemo" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
<a href="https://www.modelscope.cn/studios/liekkas/RapidOCRDemo/summary" target="_blank"><img src="https://img.shields.io/badge/ModelScope-Demo-blue"></a>
<a href="https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/assets/RapidOCRDemo.ipynb" target="_blank"><img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/colab-badge.svg" alt="Open in Colab"></a>
<a href="https://aistudio.baidu.com/aistudio/projectdetail/4444785?sUid=57084&shared=1&ts=1660896122332" target="_blank"><img src="https://img.shields.io/badge/PP-Open in AI Studio-blue.svg"></a><br/>
<a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://github.com/RapidAI/RapidOCR/graphs/contributors"><img src="https://img.shields.io/github/contributors/RapidAI/RapidOCR?color=9ea"></a>
<a href="https://pepy.tech/project/rapidocr_onnxruntime"><img src="https://static.pepy.tech/personalized-badge/rapidocr_onnxruntime?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Ort"></a>
<a href="https://pepy.tech/project/rapidocr_openvino"><img src="https://static.pepy.tech/personalized-badge/rapidocr_openvino?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Vino"></a>
<a href="https://pepy.tech/project/rapidocr_paddle"><img src="https://static.pepy.tech/personalized-badge/rapidocr_paddle?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Paddle"></a>
<a href="https://pypi.org/project/rapidocr-onnxruntime/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr-onnxruntime"></a>
<a href="https://github.com/RapidAI/RapidOCR/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidOCR?color=ccf"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

简体中文 | [English](../README.md)

</div>

<details>
    <summary>目录</summary>

- [简介](#简介)
- [Demo](#demo)
- [安装](#安装)
- [使用](#使用)
- [相关产品概览图](#相关产品概览图)
- [文档](#文档)
- [致谢](#致谢)
- [贡献者](#贡献者)
- [贡献指南](#贡献指南)
- [加入我们](#加入我们)
- [赞助](#赞助)
- [引用](#引用)
- [Star history](#star-history)
- [开源许可证](#开源许可证)
</details>

### 简介
- 💖目前已知**运行速度最快、支持最广**，完全开源免费并支持离线快速部署的多平台多语言OCR。
- **支持的语言**: 默认是中英文，其他语言识别需要自助转换。具体参考[这里](https://rapidai.github.io/RapidOCRDocs/docs/about_model/support_language/)
- **缘起**：[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)工程化不是太好，为了方便大家在各种端上进行OCR推理，我们将PaddleOCR中的模型转换为ONNX格式，使用`Python/C++/Java/C#` 将它移植到各个平台。
- **名称来源**： 轻快好省并智能。基于深度学习的OCR技术，主打人工智能优势及小模型，以速度为使命，效果为主导。
- **使用**：
  - 如果仓库下已有模型满足要求 → RapidOCR部署使用即可。
  - 不满足要求 → 基于PaddleOCR在自己数据上微调 → RapidOCR部署。
- 如果该仓库有帮助到你，还请点个小星星⭐呗！

### [Demo](https://www.modelscope.cn/studios/liekkas/RapidOCRDemo/summary)
<div align="center">
    <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/hf_demo_part.png" alt="Demo" width="100%" height="100%">
</div>


### 安装
```bash
pip install rapidocr_onnxruntime
```

### 使用
```bash
rapidocr_onnxruntime -img 1.jpg
```


### 相关产品概览图
> 下图表格中，除**测评集**部分托管在[魔搭社区](https://www.modelscope.cn/home)下，其余均在Github下，详细介绍如下：
>
> 第一行是功能介绍
>
> 第二行是对应的仓库名称，可直接去Github搜索该名称即可。

<div>
    <img src="https://github.com/RapidAI/RapidOCRDocs/releases/download/assets/RapidOCR_RelatedProjects.png">
</div>

### 文档
完整文档请移步：[docs](https://rapidai.github.io/RapidOCRDocs/docs/)

### 致谢
- 非常感谢[DeliciaLaniD](https://github.com/DeliciaLaniD)修复ocrweb中扫描动画起始位置错位问题。
- 非常感谢[zhsunlight](https://github.com/zhsunlight)关于参数化调用GPU推理的建议以及细致周到的测试。
- 非常感谢[lzh111222334](https://github.com/lzh111222334)修复python版本下rec前处理部分bug。
- 非常感谢[AutumnSun1996](https://github.com/AutumnSun1996)在[#42](https://github.com/RapidAI/RapidOCR/issues/42)中的建议。
- 非常感谢[DeadWood8](https://github.com/DeadWood8)提供了[Nuitka打包rapidocr_web的操作文档和可执行exe](https://rapidai.github.io/RapidOCRDocs/docs/install_usage/rapidocr_web/nuitka_package/)。
- 非常感谢[Loovelj](https://github.com/Loovelj)指出对文本检测框排序时顺序问题，详情参见[issue 75](https://github.com/RapidAI/RapidOCR/issues/75)。


### 贡献者
<p align="left">
  <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=RapidAI/RapidOCR" width="50%"/>
  </a>
</p>

### 贡献指南
我们感谢所有的贡献者为改进和提升 RapidOCR 所作出的努力。
- 欢迎提交请求。对于重大更改，请先打开issue讨论您想要改变的内容。
- 请确保适当更新测试。

### 加入我们
- 微信扫描以下二维码，关注**RapidAI公众号**，回复**OCR**即可加入RapidOCR微信交流群：
    <div align="center">
        <img src="https://raw.githubusercontent.com/RapidAI/.github/main/assets/RapidAI_WeChatAccount.jpg" width="25%" height="25%" align="center">
    </div>

- 欢迎加入我们的QQ群下载模型及测试程序。1群：~~887298230~~ 已满，2群：755960114 已满，3群：450338158 4群：419196348
    <div align="center">
        <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/QQ4.jpeg" width="25%" height="25%" align="center">
    </div>

- (推荐) 加入QQ频道
    <div align="center">
        <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/QQChannel.jpg" width="25%" height="25%">
    </div>

### [赞助](https://rapidai.github.io/RapidOCRDocs/docs/sponsor/)

> [!IMPORTANT]
>
> 如果您想要赞助该项目，可直接点击当前页最上面的Sponsor按钮，请写好备注(**您的Github账号名称**)，方便添加到下面赞助列表中。

|                                                                     赞助者                                                                     |                                                                               应用的产品                                                                                |
| :--------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://github.com/cuiliang" title="cuiliang"><img src="https://avatars.githubusercontent.com/u/1972649?v=4" width=65 height=65></a>  | <a href="https://getquicker.net/" title="Quicker指尖工具箱"><img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Quicker.jpg" width=65 height=65></a> |
| <a href="https://github.com/Eunsolfs" title="Eunsolfs"><img src="https://avatars.githubusercontent.com/u/53815751?v=4" width=65 height=65></a> |                                                                                    -                                                                                    |

### 引用
如果您发现该项目对您的研究有用，请考虑引用：
```bibtex
@misc{RapidOCR 2021,
    title={{Rapid OCR}: OCR Toolbox},
    author={RapidAI Team},
    howpublished = {\url{https://github.com/RapidAI/RapidOCR}},
    year={2021}
}
```

### Star history

[![Stargazers over time](https://starchart.cc/RapidAI/RapidOCR.svg)](https://starchart.cc/RapidAI/RapidOCR)


### 开源许可证
OCR模型版权归百度所有，其他工程代码版权归本仓库所有者所有。

该项目采用 [Apache 2.0 license](../LICENSE) 开源许可证。
