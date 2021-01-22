### 检测和识别模型下载：

- [链接](https://pan.baidu.com/s/1qkqWK4wRdMjqGGbzR-FyWg)
- 提取码：30jv
- 下载之后放在`models`目录即可

### 文本检测+文本识别
- 具体参见`bpocr.py`
#### 使用方法：
```python
det_model_path = 'models\ch_mobile_v1.1_det.onnx'
rec_model_path = 'models\ch_ppocr_mobile_v2.0_rec_pre_infer.onnx'
image_path = r'test_images\long1.jpg'
text_sys = TextSystem(det_model_path, rec_model_path)
dt_boxes, rec_res = text_sys(image_path)
print(dt_boxes)
print(rec_res)
img = cv2.imread(image_path)
visualize(img, dt_boxes, rec_res, image_path)
```


### 通用超轻量文本检测模型 ch_ppocr_mobile_v1.1_det

**Note: 支持任意尺寸输入**

#### 使用方法
```python
det_model_path = 'models\ch_mobile_v1.1_det.onnx'
image_path = r'test_images\det_images\1.jpg'

test_detector = TextDetector(det_model_path)

# dst_boxes: 检测到图像中的文本框坐标，ndarray格式
# (10, 4, 2)→[10个，4个坐标，每个坐标两个点]
dt_boxes, elapse, ori_im = test_detector(image_path)

cv2.imwrite('det_results.jpg', im)
print('图像已经保存在了det_results.jpg中')
```
- 即可看到检测结果图(`images/det_results.jpg`)，效果如下：

![det_result](./test_images/det_images/det_results.jpg)

### 通用超轻量文本识别模型ch_ppocr_mobile_v2_rec

**Note:  支持任意尺寸输入**

#### 使用方法：
```python
rec_model_path = r'models\ch_ppocr_mobile_v2.0_rec_pre_infer.onnx'
image_path = r'test_images\rec_images\2021-01-19_13-44-34.png'
text_recongnizer = TextRecognizer(rec_model_path)

rec_res, elapse = text_recongnizer(image_path)
print(f'识别结果：{rec_res}\tcost: {elapse}s')
```
