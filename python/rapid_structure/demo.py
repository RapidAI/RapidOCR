# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import copy
from pathlib import Path

import cv2
import numpy as np
from rapid_layout import RapidLayout


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


if __name__ == '__main__':
    model_path = 'rapid_layout/models/layout_cdla.onnx'
    layout_engine = RapidLayout(model_path)

    img_path = 'test_images/ch.png'

    img = cv2.imread(img_path)

    layout_res, elapse = layout_engine(img)

    vis_layout(img, layout_res)
    print(layout_res)
