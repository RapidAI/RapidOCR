## RapidOCR Web Demo

### 相关问题
1. 各个阶段使用的模型以及配置参数有哪些？
     - 使用模型搭配（最优组合）为：`ch_PP-OCRv3_det + ch_ppocr_mobile_v2.0_cls + ch_PP-OCRv3_rec`
     - 所有相关参数配置参见当前目录下的`config.yaml`文件
     - 其中给出了使用模型，以及具体参数，参数具体介绍参见：[Link](https://github.com/RapidAI/RapidOCR/blob/main/python/README.md#configyaml%E4%B8%AD%E5%B8%B8%E7%94%A8%E5%8F%82%E6%95%B0%E4%BB%8B%E7%BB%8D)
2. 网页上显示的推理时间可以具体解释一下吗？
    <div align="center">
        <img src="../assets/ocrweb_time.jpg" width="80%" height="80%">
    </div>

### Web方式运行
1. 安装`requirements.txt`下相关包
    ```shell
    pip install -r requirements.txt -i https://pypi.douban.com/simple/
    ```
2. 下载`resources`目录
    - 下载链接：[百度网盘](https://pan.baidu.com/s/1PTcgXG2zEgQU6A_A3kGJ3Q?pwd=jhai) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)
    - 最终目录结构如下：
        ```text
        ocrweb
        ├── README.md
        ├── config.yaml
        ├── main.py
        ├── requirements.txt
        ├── task.py
        ├── rapidocr_onnxruntime
        │   ├── __init__.py
        │   ├── ch_ppocr_v2_cls
        │   ├── ch_ppocr_v2_det
        │   ├── ch_ppocr_v2_rec
        │   └── rapid_ocr_api.py
        ├── resources
        │   ├── models
        │   │   ├── ch_PP-OCRv3_det_infer.onnx
        │   │   ├── ch_ppocr_mobile_v2.0_cls_infer.onnx
        │   │   └── ch_PP-OCRv3_rec_infer.onnx
        │   └── rec_dict
        │       └── ppocr_keys_v1.txt
        ├── static
        │   ├── css
        │   └── js
        └── templates
            └── index.html
        ```

3. 运行`main.py`
    ```shell
    python main.py
    ```
4. 打开`http://0.0.0.0:9003/`即可， enjoy it!

### 以API方式运行和调用
1. 同**Web方式运行**中步骤1
2. 同**Web方式运行**中步骤2
3. 运行`api.py`
   ```python
   python api.py
   ```

4. 发送post请求，调用
    ```python
    import ast
    import base64
    import json

    import requests


    def get_byte(img_path):
        with open(img_path, 'rb') as f:
            img_byte = base64.b64encode(f.read())
        img_str = img_byte.decode('ascii')
        return img_str


    if __name__ == '__main__':
        header = {
                "cookie": "token=code_space;",
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Connection": "keep-alive",
                "Content-Type": "application/json; charset=UTF-8",
        }

        url = 'http://localhost:9003/ocr'

        img_path = '../images/1.jpg'
        img = get_byte(img_path)
        post_json = json.dumps({'file': img})

        response = requests.post(url, data=post_json, headers=header)

        if response.status_code == 200:
            rec_res = ast.literal_eval(response.text)
            print(rec_res)
        else:
            print(response.status_code)

    ```

5. 输出以下结果，即为正确。
   ```text
   [['0', '香港深圳抽血', '0.93583983'], ['1', '专业查性别', '0.89865875'], ['2', '专业鉴定B超单', '0.9955703'], ['3', 'b超仪器查性别', '0.99489486'], ['4', '加微信eee', '0.99073666'], ['5', '可邮寄', '0.99923944']]
   ```