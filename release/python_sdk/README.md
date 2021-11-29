### sdk_rapidocr_v1.0.0

#### 说明
- whl安装时，只是确定是否安装onnxruntime，并未强制到指定版本
- 不过建议安装`onnxruntime>=1.7.0`,推理更快一些

#### 使用方法
1. 下载`sdk_rapidocr_v1.0.0`目录
2. `pip install rapidocr-1.0.0-py3-none-any.whl`
3. 下载模型和字典文件 [提取码:drf1](https://pan.baidu.com/s/103kx0ABtU7Lif57cv397oQ)
4. 运行`test_demo.py`

#### 目录结构
```text
sdk_rapidocr_v1.0.0
├── images
├── rapidocr-1.0.0-py3-none-any.whl
├── README.md
├── requirements.txt
├── resources
│   ├── models
│   │   ├── ch_mobile_v2.0_rec_infer.onnx
│   │   ├── ch_ppocr_mobile_v2.0_cls_infer.onnx
│   │   └── ch_PP-OCRv2_det_infer.onnx
│   └── ppocr_keys_v1.txt
└── test_demo.py
```