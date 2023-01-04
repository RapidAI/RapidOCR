## rapidocr-onnxruntime Package
<p>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
</p>

### 1. Install package by pypi.
```bash
$ pip install rapidocr-onnxruntime
```

### 2. Use.
```python
import cv2
from rapidocr_onnxruntime import RapidOCR

rapid_ocr = RapidOCR()

img = cv2.imread('test_images/ch_en_num.jpg')

result, elapse_list = rapid_ocr(img)
print(result)
print(elapse_list)

# result: [[dt_boxes], txt, score]
# 示例：[[左上, 右上, 右下, 左下], '小明', '0.99']

# elapse_list: [det_elapse, cls_elapse, rec_elapse]
# all_elapse = det_elapse + cls_elapse + rec_elapse
```
