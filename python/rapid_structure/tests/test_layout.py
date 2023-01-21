# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2

root_dir = Path(__file__).resolve().parent.parent

sys.path.append(str(root_dir))

from rapid_layout import RapidLayout


def test_layout():
    img_path = str(root_dir / 'test_images' / 'layout.png')
    img = cv2.imread(img_path)

    layout_engine = RapidLayout()

    layout_res, elapse = layout_engine(img)

    assert len(layout_res) == 13
