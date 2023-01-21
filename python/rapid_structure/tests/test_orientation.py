# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
import pytest

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir.parent))

from rapid_orientation import RapidOrientation
text_orientation = RapidOrientation()


@pytest.mark.parametrize(
    'img_path, result',
    [
        (f'{root_dir}/test_files/img_rot0_demo.jpg', '0'),
        (f'{root_dir}/test_files/img_rot180_demo.jpg', '180'),
    ]
)
def test_img(img_path, result):
    img = cv2.imread(str(img_path))
    pred_result, _ = text_orientation(img)

    assert pred_result == result
