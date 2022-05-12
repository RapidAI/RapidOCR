### 使用步骤
1. 下载`python/onnxruntime_infer`目录到本地
2. 下载相应模型和用于显示的字体文件
   - 网盘下对应模型可直接替换([百度网盘](https://pan.baidu.com/s/1mkirNltJS481In4g81jP3w?pwd=zy37) | [Google Drive](https://drive.google.com/drive/folders/1x_a9KpCo_1blxH1xFOfgKVkw1HYRVywY?usp=sharing))

   - 下载之后模型和相应字体文件放在`fonts`和`models`下，最终目录结构如下：
       ```text
       models
         |-- ch_PP-OCRv2_det_infer.onnx
         |-- ch_ppocr_mobile_v2.0_cls_infer.onnx
         |-- ch_ppocr_mobile_v2.0_det_infer.onnx
         |-- ch_ppocr_server_v2.0_det_infer.onnx
         |-- ch_ppocr_server_v2.0_rec_infer.onnx
         |-- en_number_mobile_v2.0_rec_infer.onnx
         |-- korean_mobile_v2.0_rec_infer.onnx
         `-- japan_rec_crnn.onnx

       fonts
         |-- msyh.ttc
         `-- korean.ttf
       ```

1. 运行
   - 接口调用方式运行
     - 如果想要使用其他语言的识别模型，可以只对`rec_model_path`和`keys_path`作对应更改即可
     - **!!!各个不同语言的推理代码区别只在于模型和字典文件!!!**
        ```python
        from rapid_ocr_api import TextSystem, visualize

        det_model_path = 'models/ch_ppocr_mobile_v2.0_det_infer.onnx'
        cls_model_path = 'models/ch_ppocr_mobile_v2.0_cls_infer.onnx'

        # 中英文识别
        rec_model_path = 'models/ch_ppocr_mobile_v2.0_rec_infer.onnx'
        keys_path = 'rec_dict/ppocr_keys_v1.txt'

        text_sys = TextSystem(det_model_path,
                              rec_model_path,
                              use_angle_cls=True,
                              cls_model_path=cls_model_path,
                              keys_path=keys_path)

        image_path = r'test_images/det_images/ch_en_num.jpg'
        dt_boxes, rec_res = text_sys(image_path)
        visualize(image_path, dt_boxes, rec_res)


        # 只有中英文和数字识别
        rec_model_path = 'models/en_number_mobile_v2.0_rec_infer.onnx'
        keys_path = 'rec_dict/en_dict.txt'

        text_sys = TextSystem(det_model_path,
                              rec_model_path,
                              use_angle_cls=True,
                              cls_model_path=cls_model_path,
                              keys_path=keys_path)

        image_path = r'test_images/det_images/en_num.png'
        dt_boxes, rec_res = text_sys(image_path)
        visualize(image_path, dt_boxes, rec_res)


        # 日语识别
        rec_model_path = 'models/japan_rec_crnn.onnx'
        keys_path = 'rec_dict/japan_dict.txt'

        text_sys = TextSystem(det_model_path,
                              rec_model_path,
                              use_angle_cls=True,
                              cls_model_path=cls_model_path,
                              keys_path=keys_path)

        image_path = r'test_images/det_images/japan.png'
        dt_boxes, rec_res = text_sys(image_path)
        visualize(image_path, dt_boxes, rec_res)


        # 韩语识别
        rec_model_path = 'models/korean_mobile_v2.0_rec_infer.onnx'
        keys_path = 'rec_dict/korean_dict.txt'
        font_path = 'fonts/korean.ttf'

        text_sys = TextSystem(det_model_path,
                              rec_model_path,
                              use_angle_cls=True,
                              cls_model_path=cls_model_path,
                              keys_path=keys_path)

        image_path = r'test_images/det_images/korean_1.jpg'
        dt_boxes, rec_res = text_sys(image_path)
        visualize(image_path, dt_boxes, rec_res, font_path)
        ```

    - 命令行运行
        ```shell
        $ bash test_demo.sh
        ```

### 相关调节参数
|参数名称|作用|建议取值范围|默认值|代码位置|
|:---:|:---:|:---:|:---:|:---:|
|box_thresh|文本检测所得框是否保留的阈值，值越大，召回率越低|[0, 1]|0.5|[text_detect.py](https://github.com/RapidAI/RapidOCR/blob/6aa79aa390c9c9e8f41df0f0c35f3dca97e6dc93/python/ch_ppocr_mobile_v2_det/text_detect.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L55)|
|unclip_ratio|控制文本检测框的大小，值越大，检测框整体越大|[1.6, 2.0]|1.6|[text_detect.py](https://github.com/RapidAI/RapidOCR/blob/6aa79aa390c9c9e8f41df0f0c35f3dca97e6dc93/python/ch_ppocr_mobile_v2_det/text_detect.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L57)|
|text_score|文本识别结果置信度，值越大，把握越大|[0, 1]|0.5|[rapidOCR.py](https://github.com/RapidAI/RapidOCR/blob/6aa79aa390c9c9e8f41df0f0c35f3dca97e6dc93/python/rapidOCR.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L270)|


### onnxruntime-gpu版推理配置

1. onnxruntime-gpu需要严格按照与cuda、cudnn版本对应来安装，具体参考[文档](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)，**这一步关乎后面是否可以成功调用GPU**
2. 推理代码中，加载onnx模型部分，用以下对应语言代码替换即可,详细参见：[官方教程](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
   - python版本
      ```python
      # 根据机器配置，安装对应版本的onnxruntime-gpu
      # pip install onnxruntime-gpu==xxxx

      import onnxruntime as ort

      model_path = '<path to model>'

      providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
      session = ort.InferenceSession(model_path, providers=providers)
      ```
   - C/C++版本
      ```c++
      OrtSessionOptions* session_options = /* ... */;

      OrtCUDAProviderOptions options;
      options.device_id = 0;
      options.arena_extend_strategy = 0;
      options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
      options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
      options.do_copy_in_default_stream = 1;

      SessionOptionsAppendExecutionProvider_CUDA(session_options, &options);
      ```
3. 推理时间粗略对比(完整跑完一张图像)

   |推理方式|推理图像数目|耗费时间|
   |:---:|:---:|:---:|
   |CPU|46张图像|191 s|
   |GPU|46张图像|52.38 s|
   ||||
