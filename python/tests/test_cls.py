# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2
import pytest
from base_module import BaseModule


@pytest.mark.parametrize(
    'package_name',
    [('rapidocr_onnxruntime'),
     ('rapidocr_openvino')]
)
def test_cls(package_name: str):
    module_name = 'ch_ppocr_v2_cls'
    class_name = 'TextClassifier'

    base = BaseModule(package_name=package_name)
    TextClassifier = base.init_module(module_name, class_name)

    yaml_path = base.package_dir / module_name / 'config.yaml'
    config = base.read_yaml(str(yaml_path))
    config['model_path'] = str(base.package_dir / config['model_path'])

    text_cls = TextClassifier(config)

    img_path = base.tests_dir / 'test_files' / 'text_cls.jpg'
    img = cv2.imread(str(img_path))
    result = text_cls([img])
    assert result[1][0][0] == '180'
