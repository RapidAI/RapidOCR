# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2
import pytest
from base_module import BaseModule


@pytest.mark.parametrize("package_name", ["rapidocr_onnxruntime", "rapidocr_openvino"])
def test_det(package_name):
    module_name = "ch_ppocr_v3_rec"
    class_name = "TextRecognizer"

    base = BaseModule(package_name)
    TextRecognizer = base.init_module(module_name, class_name)

    yaml_path = base.package_dir / module_name / "config.yaml"
    config = base.read_yaml(str(yaml_path))
    config["model_path"] = str(base.package_dir / config["model_path"])

    text_rec = TextRecognizer(config)

    img_path = base.tests_dir / "test_files" / "text_rec.jpg"
    img = cv2.imread(str(img_path))
    rec_res, elapse = text_rec(img)
    assert rec_res[0][0] == "韩国小馆"
