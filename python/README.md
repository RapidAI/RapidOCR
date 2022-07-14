## Python版RapidOCR
<p>
    <a href=""><img src="https://img.shields.io/badge/Python-3.6+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
</p>

<details open>
<summary>目录</summary>

- [Python版RapidOCR](#python版rapidocr)
  - [简介和说明](#简介和说明)
  - [pip安装快速使用](#pip安装快速使用)
  - [源码使用步骤](#源码使用步骤)
  - [`config.yaml`中常用参数介绍](#configyaml中常用参数介绍)
  - [onnxruntime-gpu版推理配置](#onnxruntime-gpu版推理配置)
  - [OpenVINO GPU推理配置](#openvino-gpu推理配置)
</details>


### 简介和说明
- **各个版本的ONNX模型下载地址：**[百度网盘](https://pan.baidu.com/s/1PTcgXG2zEgQU6A_A3kGJ3Q?pwd=jhai) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing)
- 所有常用的参数配置都在[`config.yaml`](./config.yaml)下，一目了然，更加便捷
- **目前[`config.yaml`](./config.yaml)中配置为权衡速度和准确度的最优组合。**
- 每个独立的模块下均有独立的`config.yaml`配置文件，可以单独使用
- `det`部分：
  - `det`中`mobile`和`server`版，推理代码一致，直接更改配置文件中模型路径即可
  - `det`中`v2`和`v3`两个版本，推理代码一致，直接更改配置文件中模型路径即可
    ```yaml
    Det:
        module_name: ch_ppocr_v2_det
        class_name: TextDetector
        model_path: resources/models/ch_PP-OCRv3_det_infer.onnx
    ```
- `rec`部分：
  - `rec`中`mobile`和`server`版本，推理代码一致，直接更改配置文件中模型路径即可
  - `rec`中`v2`和`v3`两个版本，共用同一个推理代码。
    - 两版本差别仅在输入shape和模型。经过测试，采用`v3 rec模型`+`[3, 48, 320]`效果最好。
    - 目前配置文件`config.yaml`中（如下所示）已经改为该最优组合。
    ```yaml
    module_name: ch_ppocr_v2_rec
    class_name: TextRecognizer
    model_path: resources/models/ch_PP-OCRv3_rec_infer.onnx

    rec_img_shape: [3, 48, 320]
    rec_batch_num: 6
    keys_path: resources/rec_dict/ppocr_keys_v1.txt
    ```
- onnxruntime和openvino调用方式如下:
    ```python
    # 基于onnxruntime引擎推理
    from rapidocr_onnxruntime import TextSystem

    # 基于openvino引擎推理
    from rapidocr_openvino import TextSystem
    ```
- 值得说明的是，基于openvino推理部分中`ch_ppocr_v2_cls`部分仍然是基于onnxruntime的，原因是openvino有bug，详情见[openvino/issue](https://github.com/openvinotoolkit/openvino/issues/11501)

### pip安装快速使用
1. 安装`rapidocr_onnxruntime`包
   ```shell
   pip install rapidocr-onnxruntime
   ```

2. 下载相关的模型和配置文件
    ```shell
    $ wget https://github.com/RapidAI/RapidOCR/releases/download/v1.0.0/required_for_whl_v1.0.0.zip

    # Download by gitee
    # wget https://gitee.com/RapidAI/RapidOCR/attach_files/1126710/download/required_for_whl_v1.0.0.zip

    $ unzip required_for_whl_v1.0.0.zip
    $ cd required_for_whl_v1.0.0
    ```
    - 最终目录结构如下：
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

3. 推理使用
```python
import cv2
from rapidocr_onnxruntime import TextSystem

text_sys = TextSystem('config.yaml')

img = cv2.imread('test_images/ch_en_num.jpg')

dt_boxes, rec_res = text_sys(img)
print(rec_res)
```

### 源码使用步骤
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
        │    │   └── ch_PP-OCRv3_rec_infer.onnx
        │    └── rec_dict
        │        └── ppocr_keys_v1.txt
        └── test_images
            ├── ch_en_num.jpg
            └── single_line_text.jpg
        ```

3. 安装运行环境
   - 基于onnxruntime推理所需环境安装：
        ```bash
        pip install onnxruntime>=1.7.0

        pip install -r requirements.txt -i https://pypi.douban.com/simple/
        ```
   - 基于openvino推理所需环境安装：
        ```bash
        # Windows端
        pip install openvino==2022.1.0

        pip install -r requirements.txt -i https://pypi.douban.com/simple/
        ```
   - Note: 在Windows端，Shapely库可能自动安装会有问题，解决方案参见[Q15](../docs/FAQ.md#q-windows系统下装完环境之后运行示例程序之后报错oserror-winerror-126-找不到指定的模組)

4. 运行示例
    - 运行单元测试
        ```bash
        cd tests
        pytest test_*.py
        ```
    - 接口调用
        ```python
        import cv2

        # 基于onnxruntime引擎推理
        from rapidocr_onnxruntime import TextSystem

        # 基于openvino引擎推理
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

### [`config.yaml`](./config.yaml)中常用参数介绍
- `Global`部分
   |    参数名称      | 取值范围   | 默认值   |                       作用                       |
   |------------: | :----------: | :-----: | :----------------------------------------------|
   | `text_score`  |    [0, 1]    |   0.5   |       文本识别结果置信度，值越大，把握越大       |
   | `use_angle_cls`  |  `bool`      |   `true`   |       是否使用文本行的方向分类       |
   | `print_verbose`  |    `bool`    |   `true`   |       是否打印各个部分耗时信息       |
   | `min_height`  |    `int`    |   30   |       图像最小高度（单位是像素）<br/>低于这个值，会跳过文本检测阶段，直接进行后续识别       |
   |`width_height_ratio`| `int`| 8| 如果输入图像的宽高比大于`width_height_ratio`，则会跳过文本检测，直接进行后续识别<br/>`width_height_ratio=-1`：不用这个参数 |

    - `min_height`是用来过滤只有一行文本的图像（如下图），这类图像不会进入文本检测模块，直接进入后续过程。

      ![](./test_images/single_line_text.jpg)

- `Det`部分
    |    参数名称      | 取值范围   | 默认值   |                       作用                       |
    | ------------: | :----------: | :-----: | :----------------------------------------------|
    |  `use_cuda`   |    `bool`     | `false` |              是否使用CUDA，加速推理              |
    |`limit_side_len`| - | 736 | 限制图像边的长度的像素值 |
    |`limit_type`| `[min, max]` | `min` | 限制图像的最小边长度还是最大边为`limit_side_len` <br/> 示例解释：当`limit_type=min`和`limit_side_len=736`时，图像最小边小于736时，<br/>会将图像最小边拉伸到736，另一边则按图像原始比例等比缩放。 |
    |  `thresh`      | [0, 1] | 0.3 | 图像中文字部分和背景部分分割阈值<br/>值越大，文字部分会越小 |
    |  `box_thresh`  |    [0, 1]    |   0.5   | 文本检测所得框是否保留的阈值，值越大，召回率越低 |
    |`max_candidates`| - | 1000 | 图像中最大可检测到的文本框数目，一般够用|
    | `unclip_ratio` |  [1.6, 2.0]  |   1.6   |   控制文本检测框的大小，值越大，检测框整体越大   |
    |`use_dilation`| `bool` | `true` | 是否使用形态学中的膨胀操作，一般采用默认值即可 |
    |`score_mode` | `string`| `fast` | `fast`是求rectangle区域的平均分数，容易造成弯曲文本漏检，`slow`是求polygon区域的平均分数，会更准确，但速度有所降低，可按需选择 |

- `Cls`部分
    |    参数名称      | 取值范围   | 默认值   |                       作用                       |
    | ------------: | :----------: | :-----: | :----------------------------------------------|
    |`cls_img_shape`| - |`[3, 48, 192]`| 输入方向分类模型的图像Shape（CHW） |
    |`cls_batch_num`| - | 6 | 批次推理的batch大小，一般采用默认值即可，太大并没有明显提速，效果还可能会差 |
    |`cls_thresh`|`[0, 1]`|0.9| 方向分类结果的置信度|
    |`label_list`| - | [0, 180] | 方向分类的标签，0°或者180°，**该参数不能动** |

- `Rec`部分
    |    参数名称      | 取值范围   | 默认值   |                       作用                       |
    | ------------: | :----------: | :-----: | :----------------------------------------------|
    |`rec_img_shape`| - |`[3, 48, 320]`| 输入文本识别模型的图像Shape（CHW） |
    |`rec_batch_num`| - | 6 | 批次推理的batch大小，一般采用默认值即可，太大并没有明显提速，效果还可能会差 |
    |`keys_path`| - | - | 文本识别模型推理所使用字典文件，始识别哪种类型文本而定（中英、日文等） |

### onnxruntime-gpu版推理配置

1. **onnxruntime-gpu**需要严格按照与cuda、cudnn版本对应来安装，具体参考[文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)，**这一步关乎后面是否可以成功调用GPU**。
   ```bash
   $ pip install onnxruntime-gpu==1.xxx
   ```
2. 更改[`config.yaml`](./config.yaml)中对应部分的参数即可，详细参数介绍参见[官方文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。
    ```yaml
    use_cuda: true
    CUDAExecutionProvider:
        device_id: 0
        arena_extend_strategy: kNextPowerOfTwo
        gpu_mem_limit: 2 * 1024 * 1024 * 1024
        cudnn_conv_algo_search: EXHAUSTIVE
        do_copy_in_default_stream: true
    ```

3. 推理情况
   1. 下载基准测试数据集（`test_images_benchmark`），放到`tests/benchmark`目录下。
        - [百度网盘](https://pan.baidu.com/s/1R4gYtJt2G3ypGkLWGwUCKg?pwd=ceuo) | [Google Drive](https://drive.google.com/drive/folders/1IIOCcUXdWa43Tfpsiy6UQJmPsZLnmgFh?usp=sharing)
        - 最终目录结构如下：
            ```text
            tests/benchmark/
                ├── benchmark.py
                ├── config_gpu.yaml
                ├── config.yaml
                └── test_images_benchmark
            ```
   2. 运行以下代码（`python`目录下运行）：
        ```shell
        # CPU
        python tests/benchmark/benchmark.py --yaml_path config.yaml

        # GPU
        python tests/benchmark/benchmark.py --yaml_path config_gpu.yaml
        ```
   3. 运行相关信息汇总：（以下仅为个人测试情况，具体情况请自行测试）
        - 来自[zhsunlight](https://github.com/zhsunlight)的测试，感谢
          - 设备型号：宏碁(Acer) 暗影骑士·威N50-N93游戏台式机
          - CPU型号：十代i5-10400F 16G 512G SSD
          - GPU型号：NVIDIA GeForce GTX 1660Super 6G
          - onnxruntime-gpu: 1.11.0
          - 耗时情况：
             |设备|总耗时(s)|平均耗时(s/img)|
             |:---:|:---:|:---:|
             |CPU|296.8841|1.18282|
             |GPU|646.14667|2.57429|
        - 来自[SWHL](https://github.com/SWHL)的测试
          - 设备型号：Docker
          - CPU型号：-
          - GPU型号：NVIDIA V100S 16G
          - onnxruntime-gpu: 1.7.0
          - 耗时情况：
             |设备|总耗时(s)|平均耗时(s/img)|
             |:---:|:---:|:---:|
             |CPU|1079.4726|4.3001|
             |GPU|525.8244|2.0989|

### OpenVINO GPU推理配置
- 官方参考文档：[docs](https://docs.openvino.ai/latest/api/ie_python_api/_autosummary/openvino.runtime.Core.html?highlight=compile_model#openvino.runtime.Core.compile_model)
- 考虑到openvino只能使用自家显卡推理，通用性不高，这里暂不作相关配置说明。