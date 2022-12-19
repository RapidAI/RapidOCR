# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import time
import copy
from pathlib import Path

import numpy as np
from rapidocr_onnxruntime import RapidOCR

from .table_matcher import TableMatch
from .table_structure import TableStructurer

root_dir = Path(__file__).resolve().parent


class RapidTable():
    def __init__(self, model_path: str = None):
        self.ocr_sys = RapidOCR()

        if model_path is None:
            model_path = str(root_dir / 'models' / 'en_ppstructure_mobile_v2_SLANet.onnx')

        self.table_structure = TableStructurer(model_path)
        self.table_matcher = TableMatch()

    def __call__(self, img):
        s = time.time()
        dt_boxes, rec_res = self._ocr(copy.deepcopy(img))

        structure_res, _ = self.table_structure(copy.deepcopy(img))
        pred_html = self.table_matcher(structure_res, dt_boxes, rec_res)

        elapse = time.time() - s
        return pred_html, elapse

    def _ocr(self, img):
        h, w = img.shape[:2]

        ocr_result, _ = self.ocr_sys(img)
        dt_boxes, rec_res, scores = list(zip(*ocr_result))
        rec_res = list(zip(rec_res, scores))

        r_boxes = []
        for box in dt_boxes:
            box = np.array(box)
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        return dt_boxes, rec_res


def main():
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--model_path', type=str,
                        default=str(root_dir / 'models' / 'en_ppstructure_mobile_v2_SLANet.onnx'))
    args = parser.parse_args()

    rapid_table = RapidTable(args.model_path)

    img = cv2.imread(args.img_path)

    table_html_str, elapse = rapid_table(img)

    print(table_html_str)


if __name__ == '__main__':
    main()
