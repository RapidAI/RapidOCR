## 通用超轻量文本检测模型 ch_ppocr_mobile_v1.1_det
**Note: 支持任意尺寸输入**
#### 第一种使用方法：
- 直接安装提供的whl包
```shell
pip install text_detect-1.0.0-py3-none-any.whl
```
- 运行
```python
from text_detect import TextDetector
import cv2

test_detector = TextDetector()

image_path = 'images/1.jpg'

# dst_boxes: 检测到图像中的文本框坐标，ndarray格式，
# e.g (10, 4, 2)→[10个，4个坐标对，每个坐标两个点]
dst_boxes, im = test_detector(image_path)

cv2.imwrite('det_results.jpg', im)
```
#### 第二种使用方法
- 下载`ch_ppocr_mobile_v1.1_det`整个文件夹到本地目录
- 运行
```shell
cd ch_mobile_det
python demo.py
```
- 即可看到检测结果图(`images/det_results.jpg`)，效果如下：

![det_result](./ch_ppocr_mobile_v1.1_det/images/det_results.jpg)