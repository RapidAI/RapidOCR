# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2
from base_module import BaseModule


class TestONNXRuntime():
    module_name = 'ch_ppocr_v3_det'
    class_name = 'TextDetector'

    base = BaseModule(package_name='rapidocr_onnxruntime')
    TextDetector = base.init_module(module_name, class_name)

    yaml_path = base.package_dir / module_name / 'config.yaml'
    config = base.read_yaml(str(yaml_path))
    config['model_path'] = str(base.package_dir / config['model_path'])

    text_det = TextDetector(config)

    def test_det(self):
        img_path = 'test_files/text_det.jpg'

        img = cv2.imread(img_path)

        dt_boxes, elapse = self.text_det(img)
        assert dt_boxes.shape == (18, 4, 2)


class TestOpenVINO():
    module_name = 'ch_ppocr_v3_det'
    class_name = 'TextDetector'

    base = BaseModule(package_name='rapidocr_openvino')
    TextDetector = base.init_module(module_name, class_name)

    yaml_path = base.package_dir / module_name / 'config.yaml'
    config = base.read_yaml(str(yaml_path))
    config['model_path'] = str(base.package_dir / config['model_path'])

    text_det = TextDetector(config)

    def test_det(self):
        img_path = 'test_files/text_det.jpg'

        img = cv2.imread(img_path)

        dt_boxes, elapse = self.text_det(img)
        assert dt_boxes.shape == (18, 4, 2)
