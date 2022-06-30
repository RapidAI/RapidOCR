## RapidOCR Web Demo

### 相关问题
1. 各个阶段使用的模型以及配置参数有哪些？
     - 所有相关参数配置参见当前目录下的`config.yaml`文件
     - 其中给出了使用模型，以及具体参数，参数具体介绍参见：[Link](https://github.com/RapidAI/RapidOCR/blob/main/python/README.md#configyaml%E4%B8%AD%E5%B8%B8%E7%94%A8%E5%8F%82%E6%95%B0%E4%BB%8B%E7%BB%8D)
2. 网页上显示的推理时间可以具体解释一下吗？
    <div align="center">
        <img src="../assets/ocrweb_time.jpg" width="80%" height="80%">
    </div>

### 运行
1. 安装`requirements.txt`下相关包
    ```shell
    pip install -r requirements.txt
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
        │   ├── ch_ppocr_v3_rec
        │   └── rapid_ocr_api.py
        ├── resources
        │   ├── models
        │   │   ├── ch_PP-OCRv2_det_infer.onnx
        │   │   ├── ch_ppocr_mobile_v2.0_cls_infer.onnx
        │   │   └── ch_ppocr_mobile_v2.0_rec_infer.onnx
        │   └── rec_dict
        │       ├── en_dict.txt
        │       ├── japan_dict.txt
        │       ├── korean_dict.txt
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
4. 打开`http://127.0.0.1:9003/`即可， enjoy this.
