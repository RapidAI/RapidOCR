<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Logov2_black.png"  width="60%" height="60%">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Logov2_white.png"  width="60%" height="60%">
    <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Logov2_white.png">
  </picture>

<div>&nbsp;</div>
<div align="center">
    <b><font size="4"><i>Open source OCR for the security of the digital world</i></font></b>
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

Join our [Discord](https://discord.gg/33eyQJq498)

[简体中文](./README-CN.md) | English
</div>

### 📝 Introduction

Introducing the foremost multi-platform, multi-lingual OCR tool that boasts unparalleled speed, expansive support, and complete openness. This exceptional software is entirely free and renowned for facilitating swift offline deployments.

**Supported Languages**: It inherently supports Chinese and English, with self-service conversion required for additional languages. Please refer [here](https://rapidai.github.io/RapidOCRDocs/main/blog/2022/09/28/%E6%94%AF%E6%8C%81%E8%AF%86%E5%88%AB%E8%AF%AD%E8%A8%80/) for specific language support details.

**Rationale**: Acknowledging the limitations in [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)'s architecture, we embarked on a mission to simplify OCR inference across diverse platforms. This endeavor culminated in converting PaddleOCR's model to the versatile ONNX format and seamlessly integrating it into Python, C++, Java, and C# environments.

**Etymology**: Derived from its essence, RapidOCR embodies lightness, velocity, affordability, and intelligence. Rooted in deep learning, this OCR technology underscores AI's prowess and emphasizes compact models, prioritizing swiftness without compromising efficacy.

**Usage Scenarios**:

- **Instant Deployment**: If the pre-existing models within our repository suffice, simply leverage RapidOCR for swift deployment.
- **Customization**: In case of specific requirements, refine PaddleOCR with your data and proceed with RapidOCR deployment, ensuring tailored results.

If our repository proves beneficial to your endeavors, kindly consider leaving a star ⭐ on GitHub to show your appreciation. It means the world to us!

### 🎥 Visualization

<div align="center">
    <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/demo.gif" alt="Demo" width="100%" height="100%">
</div>

### 🛠️ Installation

```bash
pip install rapidocr onnxruntime
```

### 📋 Usage

```python
from rapidocr import RapidOCR

engine = RapidOCR()

img_url = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/master/resources/test_files/ch_en_num.jpg"
result = engine(img_url)
print(result)

result.vis("vis_result.jpg")
```

### 📚 Documentation

Full documentation can be found on [docs](https://rapidai.github.io/RapidOCRDocs/), in Chinese.

### 👥 Who use? ([more](https://github.com/RapidAI/RapidOCR/network/dependents))

- [Docling](https://github.com/DS4SD/docling)
- [CnOCR](https://github.com/breezedeus/CnOCR)
- [api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
- [arknights-mower](https://github.com/ArkMowers/arknights-mower)
- [pensieve](https://github.com/arkohut/pensieve)
- [genshin_artifact_auxiliary](https://github.com/SkeathyTomas/genshin_artifact_auxiliary)
- [ChatLLM](https://github.com/yuanjie-ai/ChatLLM)
- [langchain](https://github.com/langchain-ai/langchain)
- [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)
- [JamAIBase](https://github.com/EmbeddedLLM/JamAIBase)
- [PAI-RAG](https://github.com/aigc-apps/PAI-RAG)
- [ChatAgent_RAG](https://github.com/junyuyang7/ChatAgent_RAG)
- [OpenAdapt](https://github.com/OpenAdaptAI/OpenAdapt)
- [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR)

> For more projects that use RapidOCR, you are welcome to [register](https://github.com/RapidAI/RapidOCR/discussions/286) at the registration address. Registration is solely for product promotion.

### 🙏 Acknowledgements

- Many thanks to [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for everything.
- Many thanks to [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch) for providing the converted PyTorch format models.
- Many thanks to [PaddleX](https://github.com/PaddlePaddle/PaddleX) for providing the document models.
- Many thanks to [DeliciaLaniD](https://github.com/DeliciaLaniD) for fixing the misplaced start position of scan animation in ocrweb.
- Many thanks to [zhsunlight](https://github.com/zhsunlight) for the suggestion about parameterized call GPU reasoning and the careful and thoughtful testing.
- Many thanks to [lzh111222334](https://github.com/lzh111222334) for fixing some bugs of rec preprocessing under python version.
- Many thanks to [AutumnSun1996](https://github.com/AutumnSun1996) for the suggestion in the [#42](https://github.com/RapidAI/RapidOCR/issues/42).
- Many thanks to [DeadWood8](https://github.com/DeadWood8) for providing the [document](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr_web/nuitka_package/) which packages rapidocr_web to exe by Nuitka.
- Many thanks to [Loovelj](https://github.com/Loovelj) for fixing the bug of sorting the text boxes. For details see [issue 75](https://github.com/RapidAI/RapidOCR/issues/75).

### 🤝 Contribution Guide

This repository contains the **Python** component of RapidOCR. Components for other languages have been migrated to separate repositories.

For the complete workflow on contributing to Python development, please refer to: [**Python CONTRIBUTING**](docs/CONTRIBUTING.md).

### 🎖 Code Contributors

<p align="left">
  <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=RapidAI/RapidOCR&max=400&columns=10" width="60%"/>
  </a>
</p>

### 🌟 Sponsors & Backers

RapidOCR is an Apache2.0-licensed open source project with its ongoing development made possible entirely by the support of these awesome backers. If you'd like to join them, please consider [sponsoring RapidOCR's development](https://rapidai.github.io/RapidOCRDocs/main/sponsor/).

#### Sponsors

|Sponsors|Application|Introduction|
|:---:|:---:|:---|
|<img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Quicker.jpg" width=65 height=65>|[Quicker](https://getquicker.net/)|Your fingertip toolbox|

#### Backers

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Eunsolfs">
        <img src="https://avatars.githubusercontent.com/u/53815751?v=4" width="60" />
      </a><br />
      <sub><a href="https://github.com/Eunsolfs">@Eunsolfs</a></sub>
    </td>
    <td align="center">
      <a href="https://github.com/youzzhang">
        <img src="https://avatars.githubusercontent.com/u/49047676?v=4" width="60" />
      </a><br />
      <sub><a href="https://github.com/youzzhang">@youzzhang</a></sub>
    </td>
  </tr>
</table>

### 📜 Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{RapidOCR 2021,
    title={{Rapid OCR}: OCR Toolbox},
    author={RapidAI Team},
    howpublished = {\url{https://github.com/RapidAI/RapidOCR}},
    year={2021}
}
```

### ⭐️ Stargazers over time

[![Stargazers over time](https://starchart.cc/RapidAI/RapidOCR.svg?variant=adaptive)](https://starchart.cc/RapidAI/RapidOCR)

### ⚖️ License

The copyright of the OCR model is held by Baidu, while the copyrights of all other engineering scripts are retained by the repository's owner.

This project is released under the [Apache 2.0 license](./LICENSE).
