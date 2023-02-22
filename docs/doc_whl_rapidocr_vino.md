## rapidocr-openvino Package
<p>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
</p>

### 1. Install package by pypi.
```shell
$ pip install rapidocr-openvino
```


### 2. Use.
- Run by script.
    ```python
    import cv2
    from rapidocr_openvino import RapidOCR

    rapid_ocr = RapidOCR()

    img_path = 'tests/test_files/ch_en_num.jpg'

    # str
    result = rapid_ocr(img_path)

    # np.ndarray
    img = cv2.imread('tests/test_files/ch_en_num.jpg')
    result = rapid_ocr(img)

    # bytes
    with open(img_path, 'rb') as f:
        result = rapid_ocr(f.read())

    # Path
    result = rapid_ocr(Path(img_path))
    print(result)

    # result: [[dt_boxes], txt, score]
    # 示例：[[左上, 右上, 右下, 左下], '小明', '0.99']

    # elapse_list: [det_elapse, cls_elapse, rec_elapse]
    # all_elapse = det_elapse + cls_elapse + rec_elapse

    # If without valid texts, result: (None, None)
    ```

- Run by command line.
  ```bash
  $ rapidocr_openvino -h
    usage: rapidocr_openvino [-h] [-img IMG_PATH] [-p]

    optional arguments:
    -h, --help            show this help message and exit
    -img IMG_PATH, --img_path IMG_PATH
    -p, --print_cost

  $ rapidocr_openvino -img tests/test_files/ch_en_num.jpg
  ```