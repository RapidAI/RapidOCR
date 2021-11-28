### 相关问题
1. 各个阶段使用的模型有哪些？
  可以从`task.py`中看到
   ```python
    det_model_path = 'resources/models/ch_PP-OCRv2_det_infer.onnx'
    cls_model_path = 'resources/models/ch_ppocr_mobile_v2.0_cls_infer.onnx'
    rec_model_path = 'resources/models/ch_ppocr_mobile_v2.0_rec_infer.onnx'
   ```
2. 网页上显示的推理时间可以具体解释一下吗？
    <div align="center">
        <img src="../assets/ocrweb_time.jpg" width="100%" height="100%">
    </div>
3. OCR各个阶段的参数值都从哪里可以看到？
    - 从各个阶段的目录下对应类的初始化函数可以看到

### 运行
1. 安装`requirements.txt`下相关包
    ```shell
    pip install -r requirements.txt
    ```
2. 相关模型
    - 下载链接：[提取码：30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)
    - 将模型放到`models`目录即可

3. 运行`main.py`
    ```shell
    python main.py
    ```
4. 打开`http://127.0.0.1:9003/`，即可

### 目录结构
```text
.
├── main.py
├── README.md
├── resources
│   ├── rapidOCR.py
│   ├── ch_ppocr_mobile_v2_cls
│   ├── ch_ppocr_mobile_v2_det
│   ├── ch_ppocr_mobile_v2_rec
│   ├── __init__.py
│   └── models
├── static
│   ├── css
│   └── js
├── task.py
└── templates
    └── index.html
```