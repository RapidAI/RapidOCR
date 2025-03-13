<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/RapidOCR_LOGO_white.png"  width="55%" height="55%">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/RapidOCR_LOGO.png"  width="55%" height="55%">
    <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/RapidOCR_LOGO.png">
  </picture>

<div>&nbsp;</div>
<div align="center">
    <b><font size="4"><i>Open source OCR for the security of the digital world</i></font></b>
</div>
<div>&nbsp;</div>

<a href="https://huggingface.co/spaces/SWHL/RapidOCRDemo" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
<a href="https://www.modelscope.cn/studios/RapidAI/RapidOCRDemo" target="_blank"><img src="https://img.shields.io/badge/魔搭-Demo-blue"></a>
<a href="https://aistudio.baidu.com/app/highcode/33121" target="_blank"><img src="https://img.shields.io/badge/百度AI%20Studio-Demo-blue"></a>
<a href="https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/assets/RapidOCRDemo.ipynb" target="_blank"><img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/colab-badge.svg" alt="Open in Colab"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.13-aff.svg"></a>
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

[简体中文](./docs/README_zh.md) | English
</div>

### Introduction

💖 Introducing the foremost multi-platform, multi-lingual OCR tool that boasts unparalleled speed, expansive support, and complete openness. This exceptional software is entirely free and renowned for facilitating swift offline deployments. Core to its efficiency is the ONNXRuntime inference engine, offering 4 to 5 times the speed of PaddlePaddle's engine while ensuring no memory leaks.

🦜 **Supported Languages**: It inherently supports Chinese and English, with self-service conversion required for additional languages. Please refer [here](https://rapidai.github.io/RapidOCRDocs/blog/2022/09/28/%E6%94%AF%E6%8C%81%E8%AF%86%E5%88%AB%E8%AF%AD%E8%A8%80/) for specific language support details.

🔎 **Rationale**: Acknowledging the limitations in [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)'s architecture, we embarked on a mission to simplify OCR inference across diverse platforms. This endeavor culminated in converting PaddleOCR's model to the versatile ONNX format and seamlessly integrating it into Python, C++, Java, and C# environments.

🎓 **Etymology**: Derived from its essence, RapidOCR embodies lightness, velocity, affordability, and intelligence. Rooted in deep learning, this OCR technology underscores AI's prowess and emphasizes compact models, prioritizing swiftness without compromising efficacy.

😉 **Usage Scenarios**:

- **Instant Deployment**: If the pre-existing models within our repository suffice, simply leverage RapidOCR for swift deployment.
- **Customization**: In case of specific requirements, refine PaddleOCR with your data and proceed with RapidOCR deployment, ensuring tailored results.

If our repository proves beneficial to your endeavors, kindly consider leaving a star ⭐ on GitHub to show your appreciation. It means the world to us!

### Visualization ([more](https://rapidai.github.io/RapidOCRDocs/visualization/))

<div align="center">
    <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/demo.gif" alt="Demo" width="100%" height="100%">
</div>

### Installation

```bash
pip install rapidocr
```

### Usage

```python
from rapidocr import RapidOCR

engine = RapidOCR()

img_url = "https://github.com/RapidAI/RapidOCR/blob/main/python/tests/test_files/ch_en_num.jpg?raw=true"
result = engine(img_url)
print(result)

result.vis()
```

### Documentation

Full documentation can be found on [docs](https://rapidai.github.io/RapidOCRDocs/), in Chinese.

### Who use?

Used by [link](https://github.com/RapidAI/RapidOCR/discussions/286)

### Acknowledgements

- Many thanks to [DeliciaLaniD](https://github.com/DeliciaLaniD) for fixing the misplaced start position of scan animation in ocrweb.
- Many thanks to [zhsunlight](https://github.com/zhsunlight) for the suggestion about parameterized call GPU reasoning and the careful and thoughtful testing.
- Many thanks to [lzh111222334](https://github.com/lzh111222334) for fixing some bugs of rec preprocessing under python version.
- Many thanks to [AutumnSun1996](https://github.com/AutumnSun1996) for the suggestion in the [#42](https://github.com/RapidAI/RapidOCR/issues/42).
- Many thanks to [DeadWood8](https://github.com/DeadWood8) for providing the [document](https://rapidai.github.io/RapidOCRDocs/install_usage/rapidocr_web/nuitka_package) which packages rapidocr_web to exe by Nuitka.
- Many thanks to [Loovelj](https://github.com/Loovelj) for fixing the bug of sorting the text boxes. For details see [issue 75](https://github.com/RapidAI/RapidOCR/issues/75).

### 🎖 Code Contributors

<p align="left">
  <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=RapidAI/RapidOCR&max=400&columns=20" width="70%"/>
  </a>
</p>

### [Sponsor](https://rapidai.github.io/RapidOCRDocs/sponsor/)

> [!IMPORTANT]
>
> If you want to sponsor the project, you can directly click the **Buy me a coffee** image, please write a note (e.g. your github account name) to facilitate adding to the sponsorship list below.
>
> <div align="left">
> <a href="https://www.buymeacoffee.com/SWHL"><img src="https://raw.githubusercontent.com/RapidAI/.github/main/assets/buymeacoffe.png" width="30%" height="30%"></a>
> </div>

|                                                                    Sponsor                                                                     |                                                                       Applied Products                                                                        |
| :-------: | :----------: |
| <a href="https://github.com/cuiliang" title="cuiliang"><img src="https://avatars.githubusercontent.com/u/1972649?v=4" width=65 height=65></a>  | <a href="https://getquicker.net/" title="Quicker"><img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Quicker.jpg" width=65 height=65></a> |
| <a href="https://github.com/Eunsolfs" title="Eunsolfs"><img src="https://avatars.githubusercontent.com/u/53815751?v=4" width=65 height=65></a> |                                                                               -                                                                               |

### Citation

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

[![Stargazers over time](https://starchart.cc/RapidAI/RapidOCR.svg)](https://starchart.cc/RapidAI/RapidOCR)

### License

The copyright of the OCR model is held by Baidu, while the copyrights of all other engineering scripts are retained by the repository's owner.

This project is released under the [Apache 2.0 license](./LICENSE).
