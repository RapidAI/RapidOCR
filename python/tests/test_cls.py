# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import base
import cv2


module_name = 'ch_ppocr_v2_cls'
class_name = 'TextClassifier'
TextClassifier = base.init_module(module_name, class_name)

yaml_path = base.root_dir / base.package_name / module_name / 'config.yaml'
config = base.read_yaml(str(yaml_path))
config['model_path'] = str(base.root_dir / config['model_path'])

text_cls = TextClassifier(config)


def test_cls():
    img_path = 'test_files/text_cls.jpg'
    img = cv2.imread(img_path)
    result = text_cls([img])
    assert result[1][0][0] == '180'
