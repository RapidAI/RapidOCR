## rapid-layout Package
<p>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pypi.org/project/rapid-layout/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapid-layout"></a>
</p>

### 1. Install package by pypi.
```bash
$ pip install rapid-layout
```

### 2. Run by script.
- RapidLayout has the default `model_path` value, you can set the different value of `model_path` to use different models, e.g. `layout_engine = RapidLayout(model_path='layout_publaynet.onnx')`
- See details, for [README_Layout](https://github.com/RapidAI/RapidOCR/blob/f133ff008a1c60edd6e0ed882da83873aa7b113a/python/rapid_structure/docs/README_Layout.md) .
- 📌 `layout.png` source: [link](https://github.com/RapidAI/RapidOCR/blob/f133ff008a1c60edd6e0ed882da83873aa7b113a/python/rapid_structure/test_images/layout.png)

```python
import cv2
from rapid_layout import RapidLayout

layout_engine = RapidLayout()

img = cv2.imread('layout.png')

layout_res, elapse = layout_engine(img)
print(layout_res)
```

### 3. Run by command line.
- Usage:
    ```bash
    $ rapid_layout -h
    usage: rapid_layout [-h] [-v] -img IMG_PATH [-m MODEL_PATH]

    optional arguments:
    -h, --help            show this help message and exit
    -v, --vis             Wheter to visualize the layout results.
    -img IMG_PATH, --img_path IMG_PATH
                            Path to image for layout.
    -m MODEL_PATH, --model_path MODEL_PATH
                            The model path used for inference.
    ```
- Example:
    ```bash
    $ rapid_layout -v -img layout.png
    ```

### 4. Result.
- Return value.
    ```python
    [
        {'bbox': array([321.4160495, 91.53214898, 562.06141263, 199.85522603]), 'label': 'text'},
        {'bbox': array([58.67292211, 107.29000663, 300.25448676, 199.68142]), 'label': 'table_caption'}
    ]
    ```
- Visualize result.
    <div align="center">
        <img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/947c6958d30f47c7c7b016f7dc308f235acec3ee/python/rapid_structure/test_images/layout_result.jpg" width="80%" height="80%">
    </div>
