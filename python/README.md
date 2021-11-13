### 运行
1. 下载相应模型和用于显示的字体文件
   - [提取码：30jv](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)
   - 下载之后放在`models`目录即可
2. 运行`rapidOCR.py`文件, 即可
   ```shell
   cd python
   python rapidOCR.py
   ```


### 文本检测+方向分类+文本识别

```python
from ch_ppocr_mobile_v2_cls import TextClassifier
from ch_ppocr_mobile_v2_det import TextDetector
from ch_ppocr_mobile_v2_rec import TextRecognizer

det_model_path = 'models/ch_ppocr_mobile_v2.0_det_infer.onnx'
cls_model_path = 'models/ch_ppocr_mobile_v2.0_cls_infer.onnx'
rec_model_path = 'models/ch_ppocr_mobile_v2.0_rec_infer.onnx'

image_path = r'test_images/det_images/1.jpg'
keys_path = 'ch_ppocr_mobile_v2_rec/ppocr_keys_v1.txt'

text_sys = TextSystem(det_model_path,
                     rec_model_path,
                     use_angle_cls=True,
                     cls_model_path=cls_model_path,
                     keys_path=keys_path)
dt_boxes, rec_res = text_sys(image_path)
visualize(image_path, dt_boxes, rec_res)
```

### 相关调节参数
|参数名称|作用|建议取值范围|默认值|代码位置|
|:---:|:---:|:---:|:---:|:---:|
|box_thresh|文本检测所得框是否保留的阈值，值越大，召回率越低|[0, 1]|0.5|[text_detect.py](https://github.com/RapidAI/RapidOCR/blob/6aa79aa390c9c9e8f41df0f0c35f3dca97e6dc93/python/ch_ppocr_mobile_v2_det/text_detect.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L55)|
|unclip_ratio|控制文本检测框的大小，值越大，检测框整体越大|[1.6, 2.0]|1.6|[text_detect.py](https://github.com/RapidAI/RapidOCR/blob/6aa79aa390c9c9e8f41df0f0c35f3dca97e6dc93/python/ch_ppocr_mobile_v2_det/text_detect.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L57)|
|text_score|文本识别结果置信度，值越大，把握越大|[0, 1]|0.5|[rapidOCR.py](https://github.com/RapidAI/RapidOCR/blob/6aa79aa390c9c9e8f41df0f0c35f3dca97e6dc93/python/rapidOCR.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L270)|


### onnxruntime-gpu版推理配置
1. 安装GPU版的onnxruntime: `pip install onnxruntime-gpu`
2. 推理代码中，加载onnx模型部分，用以下对应语言代码替换即可,详细参见：[官方教程](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
   - python版本
      ```python
      import onnxruntime as ort

      model_path = '<path to model>'

      providers = [
         ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
         }),
         'CPUExecutionProvider',
      ]

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