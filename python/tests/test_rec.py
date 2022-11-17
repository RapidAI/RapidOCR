# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2
from base_module import BaseModule


class TestONNXRuntime():
    module_name = 'ch_ppocr_v3_rec'
    class_name = 'TextRecognizer'

    base = BaseModule(package_name='rapidocr_onnxruntime')
    TextRecognizer = base.init_module(module_name, class_name)

    yaml_path = base.root_dir / base.package_name / module_name / 'config.yaml'
    config = base.read_yaml(str(yaml_path))
    config['model_path'] = str(base.root_dir / config['model_path'])

    text_rec = TextRecognizer(config)

    def test_rec(self):
        img_path = 'test_files/text_rec.jpg'
        img = cv2.imread(img_path)
        rec_res, elapse = self.text_rec([img])
        assert rec_res[0][0] == '韩国小馆'


class TestOpenVINO():
    module_name = 'ch_ppocr_v3_rec'
    class_name = 'TextRecognizer'

    base = BaseModule(package_name='rapidocr_openvino')
    TextRecognizer = base.init_module(module_name, class_name)

    yaml_path = base.root_dir / base.package_name / module_name / 'config.yaml'
    config = base.read_yaml(str(yaml_path))
    config['model_path'] = str(base.root_dir / config['model_path'])

    text_rec = TextRecognizer(config)

    def test_rec(self):
        img_path = 'test_files/text_rec.jpg'
        img = cv2.imread(img_path)
        rec_res, elapse = self.text_rec([img])
        assert rec_res[0][0] == '韩国小馆'
