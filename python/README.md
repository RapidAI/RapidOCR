## RapidOCR Python
<p>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pepy.tech/project/rapidocr_onnxruntime"><img src="https://static.pepy.tech/personalized-badge/rapidocr_onnxruntime?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Ort"></a>
    <a href="https://pepy.tech/project/rapidocr_openvino"><img src="https://static.pepy.tech/personalized-badge/rapidocr_openvino?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20Vino"></a>
</p>

<details open>
<summary>目录</summary>

- [RapidOCR Python](#rapidocr-python)
  - [简介和说明](#简介和说明)
  - [（推荐）pip安装快速使用](#推荐pip安装快速使用)
  - [源码使用步骤](#源码使用步骤)
</details>


### 简介和说明
- **各个版本的ONNX模型下载地址：**[百度网盘](https://pan.baidu.com/s/1PTcgXG2zEgQU6A_A3kGJ3Q?pwd=jhai) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)
- 所有常用的参数配置都在[`config.yaml`](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr_onnxruntime/config.yaml)下，一目了然，更加便捷
- **目前[`config.yaml`](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr_onnxruntime/config.yaml)中配置为权衡速度和准确度的最优组合。**
- 每个独立的模块下均有独立的`config.yaml`配置文件，可以单独使用
- `det`部分，`mobile` | `server` | `v2` | `v3`推理代码均为一个
  - 如果想要指定自己的模型，可以在初始化时，指定`det`模型路径，其他相关参数同理
    ```python
    from rapidocr_onnxruntime import RapidOCR
    rapid_ocr = RapidOCR(det_model_path='xxxxx.onnx')
    ```
- `rec`部分，`mobile` | `server` | `v2` | `v3`推理代码均为一个
  - `rec`中`v2`和`v3`两个版本，差别仅在输入shape和模型。经过测试，采用`v3 rec模型`+`[3, 48, 320]`效果最好
  - 目前配置文件`config.yaml`中（如下所示）已是最优组合
    ```yaml
    module_name: ch_ppocr_v3_rec
    class_name: TextRecognizer
    model_path: resources/models/ch_PP-OCRv3_rec_infer.onnx

    rec_img_shape: [3, 48, 320]
    rec_batch_num: 6
    ```
- 关于选择哪个推理引擎（onnxruntime 或者 openvino）?
    |推理引擎|推理速度更快|占用内存更少|
    |:---:|:---:|:---:|
    |onnxruntime||✓|
    |openvino|✓|存在内存不释放的问题，参见[wiki](https://github.com/RapidAI/RapidOCR/wiki/openvino%E5%85%A5%E9%97%A8)|


### （推荐）pip安装快速使用
1. 安装包（⚠️注意：两个包接口一致，只是推理引擎不同而已，默认都是在CPU上运行）
   |包名|版本|安装命令|
   |:---|:---:|:---|
   |`rapidocr_onnxruntime`|<a href="https://pypi.org/project/rapidocr-onnxruntime/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr-onnxruntime?style=flat-square"></a>|`pip install rapidocr-onnxruntime`|
   |`rapidocr_openvino`|<a href="https://pypi.org/project/rapidocr-openvino/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr-openvino?style=flat-square"></a>|`pip install rapidocr-openvino`|

1. 推理使用
    - 脚本使用：
      - ⚠️初始化RapidOCR可不提供`config.yaml`，默认使用安装目录下的`config.yaml`。如有自定义需求，可直接通过初始化参数传入。详细参数参考下面命令行部分，和`config.yaml`基本对应。
      - 输入：`Union[str, np.ndarray, bytes, Path]`
      - 输出：
          - 有值：`([[文本框坐标], 文本内容, 置信度], 推理时间)`，
            - 示例：`[[左上, 右上, 右下, 左下], '小明', '0.99'], [0.02, 0.02, 0.85]`
          - 为空：`(None, None)`
      - 示例：
        ```python
        import cv2
        from rapidocr_onnxruntime import RapidOCR
        # from rapidocr_openvino import RapidOCR

        # RapidOCR可传入参数参考下面的命令行部分
        rapid_ocr = RapidOCR()

        img_path = 'tests/test_files/ch_en_num.jpg'

        # 输入格式一：str
        result, elapse = rapid_ocr(img_path)

        # 输入格式二：np.ndarray
        img = cv2.imread('tests/test_files/ch_en_num.jpg')
        result, elapse = rapid_ocr(img)

        # 输入格式三：bytes
        with open(img_path, 'rb') as f:
            img = f.read()
        result, elapse = rapid_ocr(img)

        # 输入格式四：Path
        result, elapse = rapid_ocr(Path(img_path))
        print(result)
        ```
    - 命令行使用：
        ```bash
        $ rapidocr_onnxruntime -h
        usage: rapidocr_onnxruntime [-h] -img IMG_PATH [-p] [--text_score TEXT_SCORE]
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
                                    [--rec_img_shape REC_IMAGE_SHAPE]
                                    [--rec_batch_num REC_BATCH_NUM]

        optional arguments:
        -h, --help            show this help message and exit
        -img IMG_PATH, --img_path IMG_PATH MUST
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
        --rec_img_shape REC_IMAGE_SHAPE
        --rec_batch_num REC_BATCH_NUM

        $ rapidocr_onnxruntime -img tests/test_files/ch_en_num.jpg
        ```

### 源码使用步骤
1. 下载整个项目到本地
   ```shell
   cd RapidOCR/python
   ```

2. 下载链接下的`resources`目录（包含模型和显示的字体文件）
   - 下载链接：[Github](https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/resources.zip) | [Gitee](https://gitee.com/RapidAI/RapidOCR/releases/download/v1.1.0/resources.zip) | [百度网盘](https://pan.baidu.com/s/1PTcgXG2zEgQU6A_A3kGJ3Q?pwd=jhai) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)
   - `resources/models`下模型搭配已经为最优组合（速度和精度平衡）
        ```text
        ch_PP-OCRv3_det + ch_ppocr_mobile_v2.0_cls +  ch_PP-OCRv3_rec
        ```
   - 最终目录如下，自行比对:
        ```text
        .
        ├── README.md
        ├── demo.py
        ├── rapidocr_onnxruntime
        │   ├── __init__.py
        │   ├── ch_ppocr_v2_cls
        │   ├── ch_ppocr_v3_det
        │   ├── ch_ppocr_v3_rec
        │   ├── config.yaml
        │   ├── rapid_ocr_api.py
        │   └── models
        │        ├── ch_PP-OCRv3_det_infer.onnx
        │        ├── ch_ppocr_mobile_v2.0_cls_infer.onnx
        │        └── ch_PP-OCRv3_rec_infer.onnx
        ├── rapidocr_openvino
        │   ├── __init__.py
        │   ├── README.md
        │   ├── ch_ppocr_v2_cls
        │   ├── ch_ppocr_v3_det
        │   ├── ch_ppocr_v3_rec
        │   ├── config.yaml
        │   ├── rapid_ocr_api.py
        │   └── models
        │        ├── ch_PP-OCRv3_det_infer.onnx
        │        ├── ch_ppocr_mobile_v2.0_cls_infer.onnx
        │        └── ch_PP-OCRv3_rec_infer.onnx
        ├── requirements.txt
        └── resources
              └── fonts
                 └── FZYTK.TTF
        ```

3. 安装运行环境
   - 基于onnxruntime推理所需环境安装：
        ```bash
        pip install onnxruntime>=1.7.0
        pip install -r requirements.txt
        ```
   - 基于openvino推理所需环境安装：
        ```bash
        pip install openvino
        pip install -r requirements.txt
        ```
   - ⚠️: 在Windows端，Shapely库自动安装可能会有问题，解决方案参见[Q15](https://github.com/RapidAI/RapidOCR/blob/main/docs/FAQ.md#q-windows系统下装完环境之后运行示例程序之后报错oserror-winerror-126-找不到指定的模組)

4. 运行示例
    - 运行单元测试
        ```bash
        pytest tests/test_*.py
        ```
    - 接口调用
        ```python
        import cv2

        # 基于onnxruntime引擎推理
        from rapidocr_onnxruntime import RapidOCR

        # 基于openvino引擎推理
        # from rapidocr_openvino import RapidOCR

        rapid_ocr = RapidOCR()

        image_path = r'test_images/det_images/ch_en_num.jpg'
        img = cv2.imread(image_path)

        result = rapid_ocr(img)
        print(result)

        # result: [[文本框坐标], 文本内容, 置信度]
        # 示例：[[左上, 右上, 右下, 左下], '小明', '0.99']
        ```
    - 直接运行`demo.py`，可直接可视化查看结果。
        ```bash
        python demo.py
        ```
