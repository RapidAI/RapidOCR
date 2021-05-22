### 2021-05-22 update:
- 添加日文识别推理代码和相应模型，详情见`japan_ppocr_mobile_v2_rec`目录
- 整理代码，去除无效代码
- 每个单独模块，添加执行脚本示例

### 2021-03-24 update:
- 更改文本识别模型支持ort1.7,推理速度有着较大提升

### 检测、方向分类和识别模型下载：
- [链接](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)
- 提取码：30jv
- 下载之后放在`models`目录即可

### 文本检测+方向分类+文本识别
- 具体参见[rapidOCR.py](./rapidOCR.py)

#### 简单示例
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

### 通用超轻量文本检测模型 ch_ppocr_mobile_v2_det_train

- **Note: 支持任意尺寸输入**
- **server版文本检测只是采用了ResNet18作为backbone,其他与超轻量无区别**
- 详细见[ch_ppocr_mobile_v2_det/text_detect.py](./ch_ppocr_mobile_v2_det/text_detect.py)

#### 简单示例
```python
det_model_path = 'models/ch_ppocr_mobile_v2_det_train.onnx'
image_path = r'test_images/det_images/1.jpg'

test_detector = TextDetector(det_model_path)

img, flag = check_and_read_gif(image_path)
if not flag:
    img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"error in loading image:{image_path}")

# dst_boxes: 检测到图像中的文本框坐标，ndarray格式
# (10, 4, 2)→[10个，4个坐标，每个坐标两个点]
dt_boxes, elapse = test_detector(img)
print(dt_boxes.shape)
print(elapse)
```

### 通用超轻量方向分类器模型 ch_ppocr_mobile_v2_cls
- **Note**: 支持`0 180`两个方向分类
- 详细见[ch_ppocr_mobile_v2_cls/text_cls.py](./ch_ppocr_mobile_v2_cls/text_cls.py)

#### 简单示例
```python
text_classifier = TextClassifier(args.cls_model_dir)

img_list, cls_res, predict_time = text_classifier(args.image_dir)
for ino in range(len(img_list)):
    print(f"分类结果:{cls_res[ino]}")
```

### 通用超轻量文本识别模型 ch_ppocr_mobile_v2_rec

- **Note:  支持任意尺寸输入**
- 详细见[ch_ppocr_mobile_v2_rec/text_recognize.py](./ch_ppocr_mobile_v2_rec/text_recognize.py)


#### 简单示例：
```python
rec_model_path = r'models/ch_ppocr_mobile_v2.0_rec_pre_infer.onnx'
image_path = r'test_images/rec_images/2021-01-19_13-44-34.png'
text_recongnizer = TextRecognizer(rec_model_path)

rec_res, elapse = text_recongnizer(image_path)
print(f'识别结果：{rec_res}/tcost: {elapse}s')
```
