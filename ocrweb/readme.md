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
4. 打开`http://127.0.0.1:9005/`，即可

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