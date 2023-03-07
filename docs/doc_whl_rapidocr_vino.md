## rapidocr-openvino Package
<p>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pepy.tech/project/rapidocr_openvino"><img src="https://static.pepy.tech/personalized-badge/rapidocr_openvino?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Vino"></a>
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
    usage: rapidocr_openvino [-h] -img IMG_PATH [-p] [--text_score TEXT_SCORE]
                            [--use_angle_cls USE_ANGLE_CLS]
                            [--use_text_det USE_TEXT_DET]
                            [--print_verbose PRINT_VERBOSE]
                            [--min_height MIN_HEIGHT]
                            [--width_height_ratio WIDTH_HEIGHT_RATIO]
                            [--det_model_path DET_MODEL_PATH]
                            [--det_limit_side_len DET_LIMIT_SIDE_LEN]
                            [--det_limit_type {max,min}]
                            [--det_thresh DET_THRESH]
                            [--det_box_thresh DET_BOX_THRESH]
                            [--det_unclip_ratio DET_UNCLIP_RATIO]
                            [--det_use_dilation DET_USE_DILATION]
                            [--det_score_mode {slow,fast}]
                            [--cls_model_path CLS_MODEL_PATH]
                            [--cls_image_shape CLS_IMAGE_SHAPE]
                            [--cls_label_list CLS_LABEL_LIST]
                            [--cls_batch_num CLS_BATCH_NUM]
                            [--cls_thresh CLS_THRESH]
                            [--rec_model_path REC_MODEL_PATH]
                            [--rec_image_shape REC_IMAGE_SHAPE]
                            [--rec_batch_num REC_BATCH_NUM]

    optional arguments:
    -h, --help            show this help message and exit
    -img IMG_PATH, --img_path IMG_PATH
    -p, --print_cost

    Global:
    --text_score TEXT_SCORE
    --use_angle_cls USE_ANGLE_CLS
    --use_text_det USE_TEXT_DET
    --print_verbose PRINT_VERBOSE
    --min_height MIN_HEIGHT
    --width_height_ratio WIDTH_HEIGHT_RATIO

    Det:
    --det_model_path DET_MODEL_PATH
    --det_limit_side_len DET_LIMIT_SIDE_LEN
    --det_limit_type {max,min}
    --det_thresh DET_THRESH
    --det_box_thresh DET_BOX_THRESH
    --det_unclip_ratio DET_UNCLIP_RATIO
    --det_use_dilation DET_USE_DILATION
    --det_score_mode {slow,fast}

    Cls:
    --cls_model_path CLS_MODEL_PATH
    --cls_image_shape CLS_IMAGE_SHAPE
    --cls_label_list CLS_LABEL_LIST
    --cls_batch_num CLS_BATCH_NUM
    --cls_thresh CLS_THRESH

    Rec:
    --rec_model_path REC_MODEL_PATH
    --rec_image_shape REC_IMAGE_SHAPE
    --rec_batch_num REC_BATCH_NUM

    $ rapidocr_openvino -img tests/test_files/ch_en_num.jpg
    ```