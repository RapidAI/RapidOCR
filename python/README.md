## Python版RapidOCR
<p>
    <a href=""><img src="https://img.shields.io/badge/Python-3.6+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
</p>

- **各个版本的ONNX模型下载地址：**[百度网盘](https://pan.baidu.com/s/1PTcgXG2zEgQU6A_A3kGJ3Q?pwd=jhai) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)
- 所有常用的参数配置都在`config.yaml`下，一目了然，更加便捷
- 每个独立的模块下均有独立的`config.yaml`配置文件，可以单独使用
- `det`部分：
  - `det`中`mobile`和`server`版，共用一个推理代码，直接更改配置文件中模型路径即可
  - `det`中`v2`和`v3`两个版本代码，共用一个推理代码，直接更改配置文件中模型路径即可
    ```yaml
    Det:
        module_name: ch_ppocr_v2_det
        class_name: TextDetector
        model_path: resources/models/ch_PP-OCRv3_det_infer.onnx
    ```
- `rec`中`mobile`和`server`版本，共用一个推理代码，直接更改配置文件中模型路径即可
- onnxruntime和OpenVINO调用方式如下:
    ```python
    # 基于onnxruntime引擎推理
    from rapidocr_onnxruntime import TextSystem

    # 基于openvin引擎推理
    from rapidocr_openvino import TextSystem
    ```
- 值得说明的是，基于openvino推理部分中`ch_ppocr_v2_cls`部分仍然是基于onnxruntime的，原因是openvino有bug，详情见[openvino/issue](https://github.com/openvinotoolkit/openvino/issues/11501)

### 使用步骤
1. 下载当前下的`rapidocr_onnxruntime`/`rapidocr_openvino`目录到本地
2. 下载链接下的`resources`目录（包含模型和显示的字体文件）
   - 下载链接：[百度网盘](https://pan.baidu.com/s/1PTcgXG2zEgQU6A_A3kGJ3Q?pwd=jhai) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)
   - `resources/models`下模型搭配已经为最优组合（速度和精度平衡）
        ```text
        ch_PP-OCRv3_det + ch_ppocr_mobile_v2.0_cls +  ch_ppocr_mobile_v2.0_rec
        ```
   - 最终目录如下:
        ```text
        .
        ├── README.md
        ├── config.yaml
        ├── test_demo.py
        ├── rapidocr_onnxruntime
        │   ├── __init__.py
        │   ├── ch_ppocr_v2_cls
        │   ├── ch_ppocr_v2_det
        │   ├── ch_ppocr_v2_rec
        │   ├── ch_ppocr_v3_rec
        │   └── rapid_ocr_api.py
        ├── rapidocr_openvino
        │   ├── __init__.py
        │   ├── README.md
        │   ├── ch_ppocr_v2_cls
        │   ├── ch_ppocr_v2_det
        │   ├── ch_ppocr_v2_rec
        │   └── rapid_ocr_api.py
        ├── requirements.txt
        ├── resources
        │    ├── fonts
        │    │   └── msyh.ttc
        │    ├── models
        │    │   ├── ch_PP-OCRv3_det_infer.onnx
        │    │   ├── ch_ppocr_mobile_v2.0_cls_infer.onnx
        │    │   └── ch_ppocr_mobile_v2.0_rec_infer.onnx
        │    └── rec_dict
        │        ├── en_dict.txt
        │        ├── japan_dict.txt
        │        ├── korean_dict.txt
        │        └── ppocr_keys_v1.txt
        └── test_images
            ├── det_images
            └── rec_images
        ```
3. 安装运行环境
   - 基于onnxruntime推理所需环境安装：
        ```bash
        pip install onnxruntime>=1.7.0

        pip install -r requirements.txt -i https://pypi.douban.com/simple/
        ```
   - 基于OpenVINO推理所需环境安装：
        ```bash
        # Windows端
        pip install openvino==2022.1.0

        pip install -r requirements.txt -i https://pypi.douban.com/simple/
        ```
   - Note: 在Windows端，Shapely库可能自动安装会有问题，解决方案参见[Q15](../docs/FAQ.md#q15-装完环境之后运行python-mainpy之后报错oserror-winerror-126-找不到指定的模組)

4. 运行示例
    - 接口调用
        ```python
        import cv2

        # 基于onnxruntime引擎推理
        from rapidocr_onnxruntime import TextSystem

        # 基于OpenVINO引擎推理
        # from rapidocr_openvino import TextSystem

        config_path = 'config.yaml'
        text_sys = TextSystem(config_path)

        image_path = r'test_images/det_images/ch_en_num.jpg'
        img = cv2.imread(image_path)
        dt_boxes, rec_res = text_sys(img)
        print(rec_res)
        ```
    - 直接运行`test_demo.py`，可直接可视化查看结果。
        ```bash
        python test_demp.py
        ```
    - `rec`部分，如果想要使用`PPOCR-v3`的话，需要将`config.yaml`中的`Rec`部分改为如下:
        ```yaml
        module_name: ch_ppocr_v3_rec
        class_name: TextRecognizer
        model_path: resources/models/PPOCRv3/ch_PP-OCRv3_rec_infer.onnx

        rec_img_shape: [3, 48, 320]
        rec_batch_num: 6
        keys_path: resources/rec_dict/ppocr_keys_v1.txt
        ```

### [`config.yaml`](./config.yaml)中常用参数介绍

|    参数名称      | 建议取值范围   | 默认值   |                       作用                       |
| :------------: | :----------: | :-----: | :----------------------------------------------:|
|  `box_thresh`  |    [0, 1]    |   0.5   | 文本检测所得框是否保留的阈值，值越大，召回率越低 |
| `unclip_ratio` |  [1.6, 2.0]  |   1.6   |   控制文本检测框的大小，值越大，检测框整体越大   |
|  `text_score`  |    [0, 1]    |   0.5   |       文本识别结果置信度，值越大，把握越大       |
|   `use_cuda`   |              | `false` |              是否使用CUDA，加速推理              |

### onnxruntime-gpu版推理配置

1. **onnxruntime-gpu**需要严格按照与cuda、cudnn版本对应来安装，具体参考[文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)，**这一步关乎后面是否可以成功调用GPU**。
   ```bash
   $ pip install onnxruntime-gpu==1.xxx
   ```
2. 更改[`config.yaml`]((./config.yaml))中对应部分的参数即可，详细参数介绍参见[官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。
    ```yaml
    use_cuda: true
    CUDAExecutionProvider:
        device_id: 0
        arena_extend_strategy: kNextPowerOfTwo
        gpu_mem_limit: 2 * 1024 * 1024 * 1024
        cudnn_conv_algo_search: EXHAUSTIVE
        do_copy_in_default_stream: true
    ```

3. 推理时间粗略对比

   |推理方式|推理图像数目|耗费时间(s/张)|
   |:---:|:---:|:---:|
   |CPU|46张图像|191 s|
   |GPU|46张图像|52.38 s|