## rapidocr-onnxruntime Package

### Install package by pypi
```bash
pip install "rapidocr-onnxruntime"
```

### Install the models and config yaml.
```bash
wget https://github.com/RapidAI/RapidOCR/releases/download/v2.0.0/required_for_whl_v1.0.0.zip

# or by gitee
# wget https://gitee.com/RapidAI/RapidOCR/attach_files/1122519/download/required_for_whl_v1.0.0.zip

unzip required_for_whl_v1.0.0.zip
cd required_for_whl_v1.0.0

```

### Use by python script
```python
import cv2
from rapidocr_onnxruntime import TextSystem

text_sys = TextSystem('config.yaml')

img = cv2.imread('test_images/ch_en_num.jpg')

dt_boxes, rec_res = text_sys(img)
print(rec_res)
```