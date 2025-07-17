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

<a href="https://huggingface.co/spaces/RapidAI/RapidOCRv3" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
<a href="https://www.modelscope.cn/studios/RapidAI/RapidOCRv3.0.0/summary" target="_blank"><img src="https://img.shields.io/badge/魔搭-Demo-blue"></a>
<a href="https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/assets/RapidOCRDemo.ipynb" target="_blank"><img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/colab-badge.svg" alt="Open in Colab"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.6-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://github.com/RapidAI/RapidOCR/graphs/contributors"><img src="https://img.shields.io/github/contributors/RapidAI/RapidOCR?color=9ea"></a>
<a href="https://pepy.tech/project/rapidocr"><img src="https://static.pepy.tech/personalized-badge/rapidocr?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20rapidocr"></a>
<a href="https://pepy.tech/project/rapidocr_onnxruntime"><img src="https://static.pepy.tech/personalized-badge/rapidocr_onnxruntime?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Ort"></a>
<a href="https://pepy.tech/project/rapidocr_openvino"><img src="https://static.pepy.tech/personalized-badge/rapidocr_openvino?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Vino"></a>
<a href="https://pepy.tech/project/rapidocr_paddle"><img src="https://static.pepy.tech/personalized-badge/rapidocr_paddle?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Paddle"></a>
<a href="https://pypi.org/project/rapidocr/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr"></a>
<a href="https://github.com/RapidAI/RapidOCR/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidOCR?color=ccf"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

简体中文 | [English](../README.md)

</div>

### 简介

💖目前，我们自豪地推出了运行速度最为迅猛、兼容性最为广泛的多平台多语言OCR工具，它完全开源免费，并支持离线环境下的快速部署。

🦜 **支持语言概览：** 默认支持中文与英文识别，对于其他语言的识别需求，我们提供了便捷的自助转换方案。具体转换指南，请参见[这里](https://rapidai.github.io/RapidOCRDocs/main/blog/2022/09/28/%E6%94%AF%E6%8C%81%E8%AF%86%E5%88%AB%E8%AF%AD%E8%A8%80/)。

🔎 **项目缘起：** 鉴于[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)在工程化方面仍有进一步优化的空间，为了简化并加速在各种终端设备上进行OCR推理的过程，我们创新地将PaddleOCR中的模型转换为了高度兼容的ONNX格式，并利用Python、C++、Java、C#等多种编程语言，实现了跨平台的无缝移植，让广大开发者能够轻松上手，高效应用。

🎓 **名称寓意：** RapidOCR，这一名称蕴含着我们对产品的深刻期待——轻快（操作简便，响应迅速）、好省（资源占用低，成本效益高）并智能（基于深度学习的强大技术，精准高效）。我们专注于发挥人工智能的优势，打造小巧而强大的模型，将速度视为不懈追求，同时确保识别效果的卓越。

😉 **使用指南：**

- 直接部署：若本仓库中已提供的模型能满足您的需求，那么您只需参考[官方文档](https://rapidai.github.io/RapidOCRDocs/main/quickstart/)进行RapidOCR的部署与使用即可。
- 定制化微调：若现有模型无法满足您的特定需求，您可以在PaddleOCR的基础上，利用自己的数据进行微调，随后再将其应用于RapidOCR的部署中，实现个性化定制。

如果您发现本仓库对您的项目或学习有所助益，恳请您慷慨地给个小星星⭐，给予我们支持与鼓励！

### 效果展示

<div align="center">
    <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/demo.gif" alt="Demo" width="100%" height="100%">
</div>

### 安装

```bash
pip install rapidocr onnxruntime
```

### 使用

```python
from rapidocr import RapidOCR

engine = RapidOCR()

img_url = "https://github.com/RapidAI/RapidOCR/blob/main/python/tests/test_files/ch_en_num.jpg?raw=true"
result = engine(img_url)
print(result)

result.vis("vis_result.jpg")
```

### 文档

完整文档请移步：[docs](https://rapidai.github.io/RapidOCRDocs)

### 谁在使用？

Used by [link](https://github.com/RapidAI/RapidOCR/discussions/286)

### 致谢

- 非常感谢[DeliciaLaniD](https://github.com/DeliciaLaniD)修复ocrweb中扫描动画起始位置错位问题。
- 非常感谢[zhsunlight](https://github.com/zhsunlight)关于参数化调用GPU推理的建议以及细致周到的测试。
- 非常感谢[lzh111222334](https://github.com/lzh111222334)修复python版本下rec前处理部分bug。
- 非常感谢[AutumnSun1996](https://github.com/AutumnSun1996)在[#42](https://github.com/RapidAI/RapidOCR/issues/42)中的建议。
- 非常感谢[DeadWood8](https://github.com/DeadWood8)提供了[Nuitka打包rapidocr_web的操作文档和可执行exe](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr_web/nuitka_package/)。
- 非常感谢[Loovelj](https://github.com/Loovelj)指出对文本检测框排序时顺序问题，详情参见[issue 75](https://github.com/RapidAI/RapidOCR/issues/75)。

### 贡献者

<p align="left">
  <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=RapidAI/RapidOCR&max=400&columns=20" width="70%"/>
  </a>
</p>

### 加入我们

- 微信扫描以下二维码，关注**RapidAI公众号**，回复**OCR**即可加入RapidOCR微信交流群：
    <div align="left">
        <img src="https://raw.githubusercontent.com/RapidAI/.github/main/assets/RapidAI_WeChatAccount.jpg" width="25%" height="25%" align="center">
    </div>

- 欢迎加入我们的QQ群下载模型及测试程序。1群：~~887298230~~ 已满，2群：~~755960114~~ 已满，3群：~~450338158~~ 已满, 4群：~~419196348~~ 已满，5群：~~1032382811~~ 已满，6群：910202308
    <div align="left">
        <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/qq_group6.jpg" width="25%" height="25%" align="center">
    </div>

### [赞助](https://rapidai.github.io/RapidOCRDocs/main/sponsor/)

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

[![Stargazers over time](https://starchart.cc/RapidAI/RapidOCR.svg?variant=adaptive)](https://starchart.cc/RapidAI/RapidOCR)

### 开源许可证

OCR模型版权归百度所有，其他工程代码版权归本仓库所有者所有。

该项目采用 [Apache 2.0 license](../LICENSE) 开源许可证。
