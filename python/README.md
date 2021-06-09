### 2021-06-09 update:
- 添加字体文件到模型下载链接中

### 2021-05-22 update:
- 添加日文识别推理代码和相应模型，详情见`japan_ppocr_mobile_v2_rec`目录
- 整理代码，去除无效代码
- 每个单独模块，添加执行脚本示例

### 2021-03-24 update:
- 更改文本识别模型支持ort1.7,推理速度有着较大提升

---

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

text_sys = TextSystem(det_model_path,
                        rec_model_path,
                        use_angle_cls=True,
                        cls_model_path=cls_model_path)
dt_boxes, rec_res = text_sys(image_path)
visualize(image_path, dt_boxes, rec_res)
```
