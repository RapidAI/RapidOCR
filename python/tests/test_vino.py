# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from base_module import BaseModule

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr_openvino import LoadImageError, RapidOCR

engine = RapidOCR()
tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"
package_name = "rapidocr_openvino"


def test_without_det():
    img_path = tests_dir / "test_without_det.png"
    result, _ = engine(img_path)
    assert result[0][1] == "の持、持场所无。"
    assert len(result) == 1


def test_only_det():
    result, _ = engine(img_path, use_det=True, use_cls=False, use_rec=False)

    assert len(result) == 18
    assert result[0][0] == [5.0, 2.0]


def test_only_cls():
    img_path = tests_dir / "text_cls.jpg"
    result, _ = engine(img_path, use_det=False, use_cls=True, use_rec=False)
    assert len(result) == 1
    assert result[0][0] == "0"


def test_only_rec():
    img_path = tests_dir / "text_rec.jpg"
    result, _ = engine(img_path, use_det=False, use_cls=False, use_rec=True)
    assert len(result) == 1
    assert result[0][0] == "韩国小馆"


def test_det_rec():
    result, _ = engine(img_path, use_det=True, use_cls=False, use_rec=True)
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_cls_rec():
    img_path = tests_dir / "text_cls.jpg"
    result, _ = engine(img_path, use_det=False, use_cls=True, use_rec=True)

    assert len(result) == 1
    assert result[0][0] == "韩国小馆"


def test_det_cls_rec():
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result, _ = engine(img)
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_empty():
    img = None
    with pytest.raises(LoadImageError) as exc_info:
        engine(img)
        raise LoadImageError
    assert exc_info.type is LoadImageError


def test_zeros():
    img = np.zeros([640, 640, 3], np.uint8)
    result, _ = engine(img)
    assert result is None


def test_input_str():
    result, _ = engine(str(img_path))
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_input_bytes():
    with open(img_path, "rb") as f:
        result, _ = engine(f.read())
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_input_path():
    result, _ = engine(img_path)
    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_input_parameters():
    img_path = tests_dir / "ch_en_num.jpg"
    engine = RapidOCR(text_score=1)
    result, _ = engine(img_path)

    assert result is None


def test_input_det_parameters():
    with pytest.raises(FileNotFoundError) as exc_info:
        engine = RapidOCR(det_model_path="1.onnx")
        result, _ = engine(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_cls_parameters():
    with pytest.raises(FileNotFoundError) as exc_info:
        engine = RapidOCR(cls_model_path="1.onnx")
        result, _ = engine(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_rec_parameters():
    with pytest.raises(FileNotFoundError) as exc_info:
        engine = RapidOCR(rec_model_path="1.onnx")
        result, _ = engine(img_path)
        raise FileNotFoundError()
    assert exc_info.type is FileNotFoundError


def test_input_three_ndim_two_channel():
    img_npy = tests_dir / "two_dim_image.npy"
    image_array = np.load(str(img_npy))
    result, _ = engine(image_array)

    assert len(result) == 1
    assert result[0][1] == " TREND PLOT REPORT"


def test_input_three_ndim_one_channel():
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, 0]
    img = img[..., None]  # (H, W, 1)

    result, _ = engine(img)

    assert result[0][1] == "正品促销"
    assert len(result) == 16


def test_det():
    module_name = "ch_ppocr_v3_det"
    class_name = "TextDetector"

    base = BaseModule(package_name)
    TextDetector = base.init_module(module_name, class_name)

    yaml_path = base.package_dir / module_name / "config.yaml"
    config = base.read_yaml(str(yaml_path))
    config["model_path"] = str(base.package_dir / config["model_path"])

    text_det = TextDetector(config)
    img_path = base.tests_dir / "test_files" / "text_det.jpg"
    img = cv2.imread(str(img_path))
    dt_boxes, elapse = text_det(img)
    assert dt_boxes.shape == (18, 4, 2)


def test_cls():
    module_name = "ch_ppocr_v2_cls"
    class_name = "TextClassifier"

    base = BaseModule(package_name=package_name)
    TextClassifier = base.init_module(module_name, class_name)

    yaml_path = base.package_dir / module_name / "config.yaml"
    config = base.read_yaml(str(yaml_path))
    config["model_path"] = str(base.package_dir / config["model_path"])

    text_cls = TextClassifier(config)

    img_path = base.tests_dir / "test_files" / "text_cls.jpg"
    img = cv2.imread(str(img_path))
    result = text_cls([img])
    assert result[1][0][0] == "180"


def test_rec():
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
