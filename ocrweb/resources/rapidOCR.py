# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import copy
from pathlib import Path
import time

import cv2
import numpy as np

drop_score = 0.5

from .ch_ppocr_mobile_v2_cls import TextClassifier
from .ch_ppocr_mobile_v2_det import TextDetector
from .ch_ppocr_mobile_v2_rec import TextRecognizer


def check_and_read_gif(img_path):
    if Path(img_path).name[-3:] in ['gif', 'GIF']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            print("Cannot read {}. This gif image maybe corrupted.")
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
                      color=(0, 0, 255),
                      thickness=1)
        cv2.putText(src_im, str(i), (int(box[0][0]), int(box[0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return src_im


class TextSystem(object):
    def __init__(self, det_model_path,
                 rec_model_path,
                 use_angle_cls=False,
                 cls_model_path=None) -> None:
        super(TextSystem).__init__()
        self.text_detector = TextDetector(det_model_path)

        self.text_recognizer = TextRecognizer(rec_model_path)

        self.use_angle_cls = use_angle_cls
        if self.use_angle_cls:
            self.text_classifier = TextClassifier(cls_model_path)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    @staticmethod
    def sorted_boxes(dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

    def __call__(self, img):
        dt_boxes, det_elapse = self.text_detector(img)
        print(f"dt_boxes num: {len(dt_boxes)}, elapse: {det_elapse}")
        if dt_boxes is None or dt_boxes.size == 0:
            return None, None, img

        start_time = time.time()
        img_crop_list = []
        dt_boxes = self.sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        crop_elapse = time.time() - start_time

        if self.use_angle_cls:
            img_crop_list, _, cls_elapse = self.text_classifier(img_crop_list)
            print(f"cls num : {len(img_crop_list)}, elapse: {cls_elapse}")

        rec_res, rec_elapse = self.text_recognizer(img_crop_list)
        print(f"rec_res num: {len(rec_res)}, elapse: {rec_elapse}")

        start_time = time.time()
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        filter_elapse = time.time() - start_time

        elapse_part = [f'{det_elapse:.4f}',
                       f'{(cls_elapse+crop_elapse):.4f}',
                       f'{(rec_elapse+filter_elapse):.4f}'
        ]
        return filter_boxes, filter_rec_res, img, elapse_part
