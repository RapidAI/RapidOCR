# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2

root_dir = Path(__file__).resolve().parent.parent

sys.path.append(str(root_dir))

from rapid_table import RapidTable


def test_table():
    img_path = str(root_dir / 'test_images' / 'table.jpg')
    img = cv2.imread(img_path)

    rapid_table = RapidTable()

    table_html_str, elapse = rapid_table(img)
    assert table_html_str.count('<tr>') == 16
