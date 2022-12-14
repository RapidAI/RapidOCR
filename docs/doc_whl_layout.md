## rapid-layout Package
<p>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
</p>

### 1. Install package by pypi.
```bash
$ pip install rapidocr-layout
```

### 2. Run by script.
```python
import cv2
from rapid_layout import RapidLayout

layout_engine = RapidLayout()

img = cv2.imread('test_images/ch.png')

layout_res, elapse = layout_engine(img)
print(layout_res)
```

### 3. Run by command line.
```bash
$ rapid_layout --img_path test_images/ch.png
```

### 4. Result.
- Return value.
    ```text
    [
    {'bbox': array([321.4160495, 91.53214898, 562.06141263, 199.85522603]), 'label': 'text'},
    {'bbox': array([58.67292211, 107.29000663, 300.25448676, 199.68142]), 'label': 'table_caption'}
    ]
    ```
- Visualize result.
    <div align="center">
        <img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/947c6958d30f47c7c7b016f7dc308f235acec3ee/python/rapid_structure/test_images/layout_result.jpg" width="80%" height="80%">
    </div>
