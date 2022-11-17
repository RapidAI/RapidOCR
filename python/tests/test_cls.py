# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from base_module import BaseModule
import cv2


class TestONNXRuntime():
    module_name = 'ch_ppocr_v2_cls'
    class_name = 'TextClassifier'

    base = BaseModule(package_name='rapidocr_onnxruntime')

    TextClassifier = base.init_module(module_name, class_name)

    yaml_path = base.root_dir / base.package_name / module_name / 'config.yaml'
    config = base.read_yaml(str(yaml_path))
    config['model_path'] = str(base.root_dir / config['model_path'])

    text_cls = TextClassifier(config)

    def test_cls(self):
        img_path = 'test_files/text_cls.jpg'
        img = cv2.imread(img_path)
        result = self.text_cls([img])
        assert result[1][0][0] == '180'


class TestOpenVINO():
    module_name = 'ch_ppocr_v2_cls'
    class_name = 'TextClassifier'

    base = BaseModule(package_name='rapidocr_openvino')

    TextClassifier = base.init_module(module_name, class_name)

    yaml_path = base.root_dir / base.package_name / module_name / 'config.yaml'
    config = base.read_yaml(str(yaml_path))
    config['model_path'] = str(base.root_dir / config['model_path'])

    text_cls = TextClassifier(config)

    def test_cls(self):
        img_path = 'test_files/text_cls.jpg'
        img = cv2.imread(img_path)
        result = self.text_cls([img])
        assert result[1][0][0] == '180'
