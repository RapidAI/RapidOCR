# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import copy
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))


class RapidOCR():
    def __init__(self, config_path=str(root_dir / 'config.yaml')):
        super(RapidOCR).__init__()
        if not Path(config_path).exists():
            raise FileExistsError(f'{config_path} does not exist!')

        config = self.read_yaml(config_path)

        global_config = config['Global']
        self.print_verbose = global_config['print_verbose']
        self.text_score = global_config['text_score']
        self.min_height = global_config['min_height']
        self.width_height_ratio = global_config['width_height_ratio']

        self.use_text_det = config['Global']['use_text_det']
        if self.use_text_det:
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

    def __call__(self, img: np.ndarray, **kwargs):
        if kwargs:
            # 获得超参数
            box_thresh = kwargs.get('box_thresh', 0.5)
            unclip_ratio = kwargs.get('unclip_ratio', 1.6)
            text_score = kwargs.get('text_score', 0.5)

            # 更新超参数
            self.text_detector.postprocess_op.box_thresh = box_thresh
            self.text_detector.postprocess_op.unclip_ratio = unclip_ratio
            self.text_score = text_score

        h, w = img.shape[:2]
        if self.width_height_ratio == -1:
            use_limit_ratio = False
        else:
            use_limit_ratio = w / h > self.width_height_ratio

        if not self.use_text_det \
                or h <= self.min_height \
                or use_limit_ratio:
            dt_boxes, img_crop_list = self.get_boxes_img_without_det(img, h, w)
            det_elapse = 0.0
        else:
            dt_boxes, det_elapse = self.text_detector(img)
            if dt_boxes is None or len(dt_boxes) < 1:
                return None, None

            if self.print_verbose:
                print(f'dt_boxes num: {len(dt_boxes)}, elapse: {det_elapse}')

            dt_boxes = self.sorted_boxes(dt_boxes)
            img_crop_list = self.get_crop_img_list(img, dt_boxes)

        cls_elapse = 0.0
        if self.use_angle_cls:
            img_crop_list, _, cls_elapse = self.text_cls(img_crop_list)

            if self.print_verbose:
                print(f'cls num: {len(img_crop_list)}, elapse: {cls_elapse}')

        rec_res, rec_elapse = self.text_recognizer(img_crop_list)
        if self.print_verbose:
            print(f'rec_res num: {len(rec_res)}, elapse: {rec_elapse}')

        filter_boxes, filter_rec_res = self.filter_boxes_rec_by_score(dt_boxes,
                                                                      rec_res)
        fina_result = [[dt.tolist(), rec[0], str(rec[1])]
                       for dt, rec in zip(filter_boxes, filter_rec_res)]
        return fina_result, [det_elapse, cls_elapse, rec_elapse]

    @staticmethod
    def read_yaml(yaml_path):
        with open(yaml_path, 'rb') as f:
            data = yaml.load(f, Loader=yaml.Loader)
        return data

    @staticmethod
    def init_module(module_name, class_name):
        module_part = importlib.import_module(module_name)
        return getattr(module_part, class_name)

    def get_boxes_img_without_det(self, img, h, w):
        x0, y0, x1, y1 = 0, 0, w, h
        dt_boxes = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        dt_boxes = dt_boxes[np.newaxis, ...]
        img_crop_list = [img]
        return dt_boxes, img_crop_list

    def get_crop_img_list(self, img, dt_boxes):
        def get_rotate_crop_image(img, points):
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

        img_crop_list = []
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        return img_crop_list

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

    def filter_boxes_rec_by_score(self, dt_boxes, rec_res):
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.text_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


if __name__ == '__main__':
    text_sys = RapidOCR('config.yaml')

    import cv2
    img = cv2.imread('resources/test_images/det_images/ch_en_num.jpg')

    result = text_sys(img)
    print(result)
