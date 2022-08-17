## rapidocr-onnxruntime Package
<p>
    <a href=""><img src="https://img.shields.io/badge/Python-3.6+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
</p>

### 1. Install package by pypi.
```bash
pip install rapidocr-onnxruntime
```

### 2. Download the models and config yaml.
```bash
wget https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/required_for_whl_v1.1.0.zip

# or by gitee
# wget https://gitee.com/RapidAI/RapidOCR/releases/download/v1.1.0/required_for_whl_v1.1.0.zip

unzip required_for_whl_v1.1.0.zip
cd required_for_whl_v1.1.0
```

- The final directory of the folder:
    ```text
    required_for_whl_v1.1.0/
    ├── config.yaml
    ├── README.md
    ├── test_demo.py
    ├── resources
    │   └── models
    │       ├── ch_ppocr_mobile_v2.0_cls_infer.onnx
    │       ├── ch_PP-OCRv3_det_infer.onnx
    │       └── ch_PP-OCRv3_rec_infer.onnx
    └── test_images
        └── ch_en_num.jpg
    ```

### 3. Use.
```python
import cv2
from rapidocr_onnxruntime import TextSystem

text_sys = TextSystem('config.yaml')

img = cv2.imread('test_images/ch_en_num.jpg')

dt_boxes, rec_res = text_sys(img)
print(rec_res)
```
