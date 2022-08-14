# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import base
import cv2


module_name = 'ch_ppocr_v3_det'
class_name = 'TextDetector'
TextDetector = base.init_module(module_name, class_name)

yaml_path = base.root_dir / base.package_name / module_name / 'config.yaml'
print(yaml_path)
config = base.read_yaml(str(yaml_path))
config['model_path'] = str(base.root_dir / config['model_path'])

print(config)
text_det = TextDetector(config)


def test_det():
    img_path = 'test_files/text_det.jpg'

    img = cv2.imread(img_path)

    dt_boxes, elapse = text_det(img)
    assert dt_boxes.shape == (18, 4, 2)
