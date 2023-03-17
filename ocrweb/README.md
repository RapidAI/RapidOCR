## RapidOCR Web Demo

<p>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pypi.org/project/rapidocr-web/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr-web"></a>
    <a href="https://pepy.tech/project/rapidocr_web"><img src="https://static.pepy.tech/personalized-badge/rapidocr_web?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
</p>


- [RapidOCR Web Demo](#rapidocr-web-demo)
  - [简要说明](#简要说明)
  - [使用步骤](#使用步骤)


### 简要说明
- 该模块依赖`rapidocr_onnxruntime`库。
- 如果想要离线部署，可以先手动下载[`rapidocr_onnxruntime`](https://pypi.org/project/rapidocr-onnxruntime/#files) whl包，再手动安装[`rapidocr_web`](https://pypi.org/project/rapidocr-web/#files) whl包来使用。
- 所用模型组合（最优组合）为：
  ```text
  ch_PP-OCRv3_det + ch_ppocr_mobile_v2.0_cls + ch_PP-OCRv3_rec
  ```
- **在线demo运行机器配置:** `4核 AMD EPYC 7K62 48-Core Processor`
- 网页上显示的推理时间具体解释如下：

    <div align="center">
        <img src="https://github.com/RapidAI/RapidOCR/blob/main/assets/ocrweb_time.jpg" width="80%" height="80%">
    </div>

### 使用步骤
1. 安装`rapidocr_web`
   ```bash
   pip install rapidocr_web
   ```

2. 运行
   - 用法:
       ```bash
       $ rapidocr_web -h
       usage: rapidocr_web [-h] [-ip IP] [-p PORT] [-api]

       optional arguments:
       -h, --help            show this help message and exit
       -ip IP, --ip IP       IP Address
       -p PORT, --port PORT  IP port
       -api, --is_api        Whether to use the api format.
       ```
   - 示例:
       ```bash
       # 界面模式
       $ rapidocr_web -ip "0.0.0.0" -p 9003

       # API调用模式
       $ rapidocr_web -ip "0.0.0.0" -p 9003 -api
       ```

3. 使用
   - 界面模式：浏览器打开`http://localhost:9003/`，enjoy it.
   - API模式:
        ```python
        # python 示例，本质就是发送一个POST请求，其他语言同理。
        import ast
        import base64
        import json

        import requests


        def get_json_format(img_path):
            with open(img_path, 'rb') as f:
                img_byte = base64.b64encode(f.read())
            img_json = json.dumps({'file': img_byte.decode('ascii')})
            return img_json


        if __name__ == '__main__':
            url = 'http://localhost:9003/ocr'
            header = {'Content-Type': 'application/json; charset=UTF-8'}

            img_path = '../images/1.jpg'
            img_json = get_json_format(img_path)

            response = requests.post(url, data=img_json, headers=header)
            if response.status_code == 200:
                rec_res = ast.literal_eval(response.text)
                print(rec_res)
            else:
                print(response.status_code)
        ```

4. API输出
   - 示例结果：
        ```text
        [
            [[[265.0, 18.0], [472.0, 231.0], [431.0, 271.0], [223.0, 59.0]], '香港深圳抽血', 0.8021483932222638],
            [[[388.0, 15.0], [636.0, 257.0], [587.0, 307.0], [339.0, 65.0]], '专业查性别', 0.7488822937011719],
            [[[215.0, 84.0], [509.0, 413.0], [453.0, 463.0], [159.0, 134.0]], '专业鉴定B超单', 0.8711239919066429],
            [[[128.0, 135.0], [430.0, 478.0], [366.0, 534.0], [64.0, 192.0]], 'b超仪器查性别', 0.8705329671502113],
            [[[58.0, 189.0], [268.0, 450.0], [209.0, 498.0], [0.0, 236.0]], '加微信eee', 0.8492027946880886],
            [[[493.0, 261.0], [617.0, 384.0], [577.0, 423.0], [454.0, 300.0]], '可邮寄', 0.7494295984506607]
        ]
        ```
   - 输出结果说明：如果图像中存在文字，则会输出`List`类型，具体格式介绍如下：
        ```text
        [
            # 坐标为左上角 → 右上角 → 右下角 → 左下角
            [[[left, top], [right, top], [right, bottom], [left, bottom]], 识别文本, 置信度]
        ]
        ```
   - 如果没有检测到文字，则会输出空列表(`[]`)。

5. **!!说明：OCR的输出结果为最原始结果，大家可按需进一步扩展。**
