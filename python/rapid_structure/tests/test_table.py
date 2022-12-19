# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2

# rapid_structure路径
root_dir = Path(__file__).resolve().parent.parent

sys.path.append(str(root_dir))

from rapid_table import RapidTable


def test_table():
    rapid_table = RapidTable()

    img = cv2.imread('test_images/table.jpg')

    table_html_str, elapse = rapid_table(img)
    assert table_html_str.count('<tr>') == 16
