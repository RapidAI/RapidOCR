## RapidOCR Web Demo

<p>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.6,<3.12-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pypi.org/project/rapidocr-web/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr-web"></a>
    <a href="https://pepy.tech/project/rapidocr_web"><img src="https://static.pepy.tech/personalized-badge/rapidocr_web?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
</p>


- [RapidOCR Web Demo](#rapidocr-web-demo)
  - [简要说明](#简要说明)
  - [桌面版使用教程](#桌面版使用教程)
  - [界面版使用步骤](#界面版使用步骤)


### 简要说明
- 该模块依赖`rapidocr_onnxruntime`库。
- 如果想要离线部署，可以先手动下载[`rapidocr_onnxruntime`](https://pypi.org/project/rapidocr-onnxruntime/#files) whl包，再手动安装[`rapidocr_web`](https://pypi.org/project/rapidocr-web/#files) whl包来使用。
- 所用模型组合（最优组合）为：
  ```text
  ch_PP-OCRv3_det + ch_ppocr_mobile_v2.0_cls + ch_PP-OCRv3_rec
  ```
- 网页上显示的推理时间具体解释如下：

    <div align="center">
        <img src="https://github.com/RapidAI/RapidOCR/blob/ae529c2ba79e6cbf04c54caf2d24feb75e947ca4/assets/ocrweb_time.jpg" width="80%" height="80%">
    </div>

### [桌面版使用教程](https://github.com/RapidAI/RapidOCR/wiki/%5BRapidOCRWeb%5D-%E6%A1%8C%E9%9D%A2%E7%89%88%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B)

### 界面版使用步骤
1. 安装`rapidocr_web`
   ```bash
   $ pip install rapidocr_web
   ```
2. 运行
   - 用法:
       ```bash
       $ rapidocr_web -h
       usage: rapidocr_web [-h] [-ip IP] [-p PORT]

       optional arguments:
       -h, --help            show this help message and exit
       -ip IP, --ip IP       IP Address
       -p PORT, --port PORT  IP port
       ```
   - 示例:
       ```bash
       $ rapidocr_web -ip 0.0.0.0 -p 9003
       ```
3. 使用
   - 浏览器打开`http://localhost:9003/`，enjoy it.
