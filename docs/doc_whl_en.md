## rapidocr-onnxruntime Package

### Install package by pypi.
```bash
pip install rapidocr-onnxruntime
```

### Download the models and config yaml.
```bash
wget https://github.com/RapidAI/RapidOCR/releases/download/v1.0.0/required_for_whl_v1.0.0.zip

# or by gitee 
# wget https://gitee.com/RapidAI/RapidOCR/attach_files/1126710/download/required_for_whl_v1.0.0.zip

unzip required_for_whl_v1.0.0.zip
cd required_for_whl_v1.0.0

```
- The final directory of the folder:
```text
required_for_whl_v1.0.0/
├── config.yaml
├── README.md
├── test_demo.py
├── resources
│   ├── models
│   │   ├── ch_ppocr_mobile_v2.0_cls_infer.onnx
│   │   ├── ch_PP-OCRv3_det_infer.onnx
│   │   └── ch_PP-OCRv3_rec_infer.onnx
│   └── rec_dict
│       └── ppocr_keys_v1.txt
└── test_images
    └── ch_en_num.jpg
```

### Use by python script.
```python
import cv2
from rapidocr_onnxruntime import TextSystem

text_sys = TextSystem('config.yaml')

img = cv2.imread('test_images/ch_en_num.jpg')

dt_boxes, rec_res = text_sys(img)
print(rec_res)
```
