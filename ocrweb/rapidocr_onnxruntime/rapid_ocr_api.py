# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import copy
import importlib
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))


class TextSystem(object):
    def __init__(self, config_path=str(root_dir / 'config.yaml')):
        super(TextSystem).__init__()
        config = self.read_yaml(config_path)

        self.print_verbose = config['Global']['print_verbose']
        self.text_score = config['Global']['text_score']

        TextDetector = self.init_module(config['Det']['module_name'],
                                        config['Det']['class_name'])
        self.text_detector = TextDetector(config['Det'])

        TextRecognizer = self.init_module(config['Rec']['module_name'],
                                          config['Rec']['class_name'])
        self.text_recognizer = TextRecognizer(config['Rec'])

        self.use_angle_cls = config['Global']['use_angle_cls']
        if self.use_angle_cls:
            TextClassifier = self.init_module(config['Cls']['module_name'],
                                              config['Cls']['class_name'])
            self.text_cls = TextClassifier(config['Cls'])

    @staticmethod
    def read_yaml(yaml_path):
        with open(yaml_path, 'rb') as f:
            data = yaml.load(f, Loader=yaml.Loader)
        return data

    @staticmethod
    def init_module(module_name, class_name):
        module_part = importlib.import_module(module_name)
        return getattr(module_part, class_name)

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

    def __call__(self, img: np.ndarray):
        dt_boxes, det_elapse = self.text_detector(img)
        if self.print_verbose:
            print(f'dt_boxes num: { len(dt_boxes)}, elapse: {det_elapse}')

        if dt_boxes is None or len(dt_boxes) < 1:
            return None, None, img, None

        start_time = time.time()
        img_crop_list = []
        dt_boxes = self.sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        crop_elapse = time.time() - start_time

        if self.use_angle_cls:
            img_crop_list, _, cls_elapse = self.text_cls(img_crop_list)
            if self.print_verbose:
                print(f'cls num: {len(img_crop_list)}, elapse: {cls_elapse}')

        rec_res, rec_elapse = self.text_recognizer(img_crop_list)
        if self.print_verbose:
            print(f'rec_res num: {len(rec_res)}, elapse: {rec_elapse}')

        start_time = time.time()
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.text_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)

        filter_elapse = time.time() - start_time
        elapse_part = [f'{det_elapse:.4f}',
                       f'{(cls_elapse+crop_elapse):.4f}',
                       f'{(rec_elapse+filter_elapse):.4f}'
        ]
        return filter_boxes, filter_rec_res, img, elapse_part


if __name__ == '__main__':
    text_sys = TextSystem('config.yaml')

    import cv2
    img = cv2.imread('resources/test_images/det_images/ch_en_num.jpg')

    result = text_sys(img)
    print(result)
