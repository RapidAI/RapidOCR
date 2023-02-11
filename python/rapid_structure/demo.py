# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import copy
from pathlib import Path

import cv2
import numpy as np

from rapid_layout import RapidLayout
from rapid_orientation import RapidOrientation
from rapid_table import RapidTable


def vis_layout(img: np.ndarray, layout_res: list) -> None:
    tmp_img = copy.deepcopy(img)
    for v in layout_res:
        bbox = np.round(v['bbox']).astype(np.int32)
        label = v['label']

        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])

        cv2.rectangle(tmp_img, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(tmp_img, label, start_point,
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    draw_img_save = Path("./inference_results/")
    if not draw_img_save.exists():
        draw_img_save.mkdir(parents=True, exist_ok=True)

    image_save = str(draw_img_save / 'layout_result.jpg')
    cv2.imwrite(image_save, tmp_img)
    print(f'The infer result has saved in {image_save}')


def vis_table(table_res):
    style_res = '''<style>td {border-left: 1px solid;border-bottom:1px solid;}
                   table, th {border-top:1px solid;font-size: 10px;
                   border-collapse: collapse;border-right: 1px solid;}
                </style>'''
    prefix_table, suffix_table = table_res.split('<body>')
    new_table_res = f'{prefix_table}{style_res}<body>{suffix_table}'

    draw_img_save = Path("./inference_results/")
    if not draw_img_save.exists():
        draw_img_save.mkdir(parents=True, exist_ok=True)

    html_path = str(draw_img_save / 'table_result.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(new_table_res)
    print(f'The infer result has saved in {html_path}')


def demo_layout():
    layout_engine = RapidLayout()

    img = cv2.imread('test_images/layout.png')

    layout_res, _ = layout_engine(img)

    vis_layout(img, layout_res)
    print(layout_res)


def demo_table():
    table_engine = RapidTable()
    img = cv2.imread('test_images/table.jpg')
    table_html_str, _ = table_engine(img)

    vis_table(table_html_str)
    print(table_html_str)


def demo_orientation():
    orientation_engine = RapidOrientation()
    img = cv2.imread('tests/test_files/img_rot180_demo.jpg')
    cls_result, _ = orientation_engine(img)
    print(cls_result)


if __name__ == '__main__':
    demo_layout()
    # demo_table()
    # demo_orientation()
