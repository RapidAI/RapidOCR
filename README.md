<div align="center">
  <img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/RapidOCR_LOGO.png" width="55%" height="55%"/>

<div>&nbsp;</div>
<div align="center">
    <b><font size="4"><i>Open source OCR for the security of the digital world</i></font></b>
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
<a href="https://pepy.tech/project/rapidocr_openvino"><img src="https://static.pepy.tech/personalized-badge/rapidocr_openvino?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Ort"></a>
<a href="https://pepy.tech/project/rapidocr_paddle"><img src="https://static.pepy.tech/personalized-badge/rapidocr_paddle?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Paddle"></a>
<a href="https://pypi.org/project/rapidocr-onnxruntime/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr-onnxruntime"></a>
<a href="https://github.com/RapidAI/RapidOCR/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidOCR?color=ccf"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

[简体中文](./docs/README_zh.md) | English
</div>

<details>
    <summary>Contents</summary>

- [Introduction](#introduction)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Acknowledgements](#acknowledgements)
- [Code Contributors](#code-contributors)
- [Contributing](#contributing)
- [Sponsor](#sponsor)
- [Citation](#citation)
- [Stargazers over time](#stargazers-over-time)
- [License](#license)

</details>

### Introduction
- The fastest running, most widely supported, completely open source and free multi-platform, multi-language OCR known to support rapid offline deployment.
- **Supported Languages**: The default is Chinese and English, other language recognition requires self-service conversion. For specific reference [here](https://rapidai.github.io/RapidOCRDocs/docs/about_model/support_language/).
- **Cause**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) is not well engineered, and to make it easier for people to do OCR inference on various ends, we converted the model in PaddleOCR to ONNX format and ported it to various platforms using `Python/C++/Java/Swift/C#`.
- **Name Source**: Light, fast, economical and smart. OCR technology based on deep learning technology focuses on artificial intelligence advantages and small models, with speed as the mission and effect as the leading role.
- **Usage**:
  - If the existing model in the repo meets the requirements → RapidOCR deployment can be used.
  - Not meeting requirements → Based on PaddleOCR. Fine-tune your own data → RapidOCR deployment.
-If this repo is helpful to you, please click on a small star ⭐ Bah!


### [Demo](https://huggingface.co/spaces/SWHL/RapidOCRDemo)

<div align="center">
    <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/hf_demo_part.png" alt="Demo" width="100%" height="100%">
</div>

### Installation
```bash
pip install rapidocr_onnxruntime
```

### Usage
```bash
rapidocr_onnxruntime -img 1.jpg
```

### Documentation
Full documentation can be found on [docs](https://rapidai.github.io/RapidOCRDocs/docs/), in Chinese.

### Acknowledgements
- Many thanks to [DeliciaLaniD](https://github.com/DeliciaLaniD) for fixing the misplaced start position of scan animation in ocrweb.
- Many thanks to [zhsunlight](https://github.com/zhsunlight) for the suggestion about parameterized call GPU reasoning and the careful and thoughtful testing.
- Many thanks to [lzh111222334](https://github.com/lzh111222334) for fixing some bugs of rec preprocessing under python version.
- Many thanks to [AutumnSun1996](https://github.com/AutumnSun1996) for the suggestion in the [#42](https://github.com/RapidAI/RapidOCR/issues/42).
- Many thanks to [DeadWood8](https://github.com/DeadWood8) for providing the [document]((https://github.com/RapidAI/RapidOCR/wiki/Nuitka%E6%89%93%E5%8C%85rapidocr_web%E6%93%8D%E4%BD%9C%E6%96%87%E6%A1%A3)) which packages rapidocr_web to exe by Nuitka.
- Many thanks to [Loovelj](https://github.com/Loovelj) for fixing the bug of sorting the text boxes. For details see [issue 75](https://github.com/RapidAI/RapidOCR/issues/75).

### Code Contributors
<p align="left">
  <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=RapidAI/RapidOCR" width="50%"/>
  </a>
</p>

### Contributing
- Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
- Please make sure to update tests as appropriate.

### [Sponsor](https://rapidai.github.io/RapidOCRDocs/docs/sponsor/)

> [!IMPORTANT]
>
> If you want to sponsor the project, you can directly click the **Buy me a coffee** image, please write a note (e.g. your github account name) to facilitate adding to the sponsorship list below.
>
> <div align="left">
>   <a href="https://www.buymeacoffee.com/SWHL"><img src="https://raw.githubusercontent.com/RapidAI/.github/main/assets/buymeacoffe.png" width="30%" height="30%"></a>
> </div>

|Sponsor|Applied Products|
|:---:|:---:|
|<a href="https://github.com/cuiliang" title="cuiliang"><img src="https://avatars.githubusercontent.com/u/1972649?v=4" width=65 height=65></a>|<a href="https://getquicker.net/" title="Quicker"><img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Quicker.jpg" width=65 height=65></a>|
|<a href="https://github.com/Eunsolfs" title="Eunsolfs"><img src="https://avatars.githubusercontent.com/u/53815751?v=4" width=65 height=65></a>| - |


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

### Stargazers over time

[![Stargazers over time](https://starchart.cc/RapidAI/RapidOCR.svg)](https://starchart.cc/RapidAI/RapidOCR)


### License
The copyright of the OCR model belongs to Baidu, and the copyright of other engineering codes belongs to the owner of this repo.

This project is released under the [Apache 2.0 license](./LICENSE).
