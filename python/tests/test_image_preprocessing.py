# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from types import SimpleNamespace

import numpy as np

import rapidocr.main as main_module
from rapidocr import RapidOCR
from rapidocr.ch_ppocr_det import TextDetOutput


def test_preprocess_img_returns_original_when_disabled(monkeypatch):
    engine = RapidOCR.__new__(RapidOCR)
    engine.cfg = SimpleNamespace(
        Global=SimpleNamespace(use_preprocess_img=False)
    )
    ori_img = np.zeros((20, 40, 3), dtype=np.uint8)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("resize_image_within_bounds should not be called")

    monkeypatch.setattr(main_module, "resize_image_within_bounds", fail_if_called)

    img, op_record = engine.preprocess_img(ori_img)

    assert img is ori_img
    assert op_record == {"preprocess": {"ratio_h": 1.0, "ratio_w": 1.0}}


def test_detect_and_crop_skips_vertical_padding_when_disabled(monkeypatch):
    engine = RapidOCR.__new__(RapidOCR)
    engine.cfg = SimpleNamespace(
        Global=SimpleNamespace(use_vertical_padding=False)
    )
    img = np.zeros((20, 40, 3), dtype=np.uint8)
    boxes = np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]], dtype=np.float32)
    detected_images = []

    def fail_if_called(*args, **kwargs):
        raise AssertionError("apply_vertical_padding should not be called")

    def text_det(det_img):
        detected_images.append(det_img)
        return TextDetOutput(boxes=boxes)

    monkeypatch.setattr(main_module, "apply_vertical_padding", fail_if_called)
    engine.text_det = text_det
    engine.crop_text_regions = lambda crop_img, det_boxes: [crop_img]
    op_record = {}

    cropped_imgs, det_res = engine.detect_and_crop(img, op_record)

    assert detected_images[0] is img
    assert cropped_imgs[0] is img
    assert det_res.boxes is boxes
    assert op_record["padding_1"] == {"top": 0, "left": 0}
