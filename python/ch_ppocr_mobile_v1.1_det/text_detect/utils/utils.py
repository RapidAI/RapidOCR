# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: utils.py
# @Time: 2021/01/16 12:00:01
# @Author: Max
import copy
from pathlib import Path

import cv2
import numpy as np


def check_and_read_gif(img_path):
    if Path(img_path).suffix in ['.gif', '.GIF']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            print(f"Cannot read {str(img_path)}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True
    return None, False


def draw_text_det_res(dt_boxes, raw_im):
    src_im = copy.deepcopy(raw_im)
    for i, box in enumerate(dt_boxes):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True,
                      color=(255, 255, 0),
                      thickness=2)
        cv2.putText(src_im, str(i), (int(box[0][0]), int(box[0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return src_im

