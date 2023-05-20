## RapidOCR Web Demo

<p>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pypi.org/project/rapidocr-web/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr-web"></a>
    <a href="https://pepy.tech/project/rapidocr_web"><img src="https://static.pepy.tech/personalized-badge/rapidocr_web?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
</p>


- [RapidOCR Web Demo](#rapidocr-web-demo)
  - [简要说明](#简要说明)
  - [桌面版使用教程](#桌面版使用教程)
  - [界面版使用步骤](#界面版使用步骤)
  - [API版使用步骤](#api版使用步骤)


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

### API版使用步骤
1. 安装`rapidocr_web[api]`
   ```bash
   $ pip install rapidocr_web[api]
   ```
2. 运行
   - 用法:
       ```bash
       $ rapidocr_api -h
       usage: rapidocr_api [-h] [-ip IP] [-p PORT]

       optional arguments:
       -h, --help            show this help message and exit
       -ip IP, --ip IP       IP Address
       -p PORT, --port PORT  IP port
       ```
   - 示例:
       ```bash
       $ rapidocr_api -ip 0.0.0.0 -p 9003
       ```
3. 使用（本质就是发送一个POST请求，其他语言同理）
    - 以文件的方式发送请求
        ```python
        import requests

        url = 'http://localhost:9003/ocr'
        img_path = '../python/tests/test_files/ch_en_num.jpg'

        with open(img_path, 'rb') as f:
            file_dict = {'image_file': (img_path, f, 'image/png')}
            response = requests.post(url, files=file_dict, timeout=60)

        print(response.json())
        ```
    - 以`base64`方式发送post请求
        ```python
        import base64
        import requests

        url = 'http://localhost:9003/ocr'
        img_path = '../python/tests/test_files/ch_en_num.jpg'

        with open(img_path, 'rb') as fa:
            img_str = base64.b64encode(fa.read())

        payload = {'image_data': img_str}
        resp = requests.post(url, data=payload)

        print(resp.json())
        ```
4. API输出
   - 示例结果：
        <details>
        <summary>详情</summary>

       ```json
        {
            "0": {
                "rec_txt": "香港深圳抽血，",
                "dt_boxes": [
                    [265, 18],
                    [472, 231],
                    [431, 271],
                    [223, 59]
                ],
                "score": "0.8175641223788261"
            },
            "1": {
                "rec_txt": "专业查性别",
                "dt_boxes": [
                    [388, 15],
                    [636, 257],
                    [587, 307],
                    [339, 65]
                ],
                "score": "0.8293875356515249"
            },
            "2": {
                "rec_txt": "专业鉴定B超单",
                "dt_boxes": [
                    [215, 84],
                    [509, 413],
                    [453, 463],
                    [159, 134]
                ],
                "score": "0.8626169338822365"
            },
            "3": {
                "rec_txt": "b超仪器查性别",
                "dt_boxes": [
                    [128, 135],
                    [430, 478],
                    [366, 534],
                    [64, 192]
                ],
                "score": "0.8449362441897392"
            },
            "4": {
                "rec_txt": "加微信eee",
                "dt_boxes": [
                    [58, 189],
                    [268, 450],
                    [209, 498],
                    [0, 236]
                ],
                "score": "0.8176911813872201"
            },
            "5": {
                "rec_txt": "可邮寄",
                "dt_boxes": [
                    [493, 261],
                    [617, 384],
                    [577, 423],
                    [454, 300]
                ],
                "score": "0.7494261413812637"
            }
        }
       ```
       </details>

   - 输出结果说明：
     - 如果图像中存在文字，则会输出json格式，具体格式介绍如下：
       ```python
       {
        "0": {
            "rec_txt": "香港深圳抽血，",  # 识别的文本
            "dt_boxes": [  # 依次为左上角 → 右上角 → 右下角 → 左下角
                [265, 18],
                [472, 231],
                [431, 271],
                [223, 59]
            ],
            "score": "0.8175641223788261"  # 置信度
            }
       }
       ```
     - 如果没有检测到文字，则会输出空json(`{}`)。
5. **!!说明：OCR的输出结果为最原始结果，大家可按需进一步扩展。**
