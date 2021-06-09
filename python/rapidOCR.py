# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import copy
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

drop_score = 0.5


def str2bool(v):
    return v.lower() in ("true", "t", "1")


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


def draw_ocr_box_txt(image, boxes, txts,
                     scores=None, drop_score=0.5,
                     font_path="./models/msyh.ttc"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def visualize(image_path, boxes, rec_res):
    image = Image.open(image_path)
    txts = [rec_res[i][0] for i in range(len(rec_res))]
    scores = [rec_res[i][1] for i in range(len(rec_res))]

    draw_img = draw_ocr_box_txt(image, boxes,
                                txts, scores,
                                drop_score=0.5)

    draw_img_save = Path("./inference_results/")
    if not draw_img_save.exists():
        draw_img_save.mkdir(parents=True,
                            exist_ok=True)

    image_save = str(draw_img_save / f'infer_{Path(image_path).name}')
    cv2.imwrite(image_save, draw_img[:, :, ::-1])
    print(f'The infer result has saved in {image_save}')


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

    @staticmethod
    def load_image(image_path):
        img, flag = check_and_read_gif(image_path)
        if not flag:
            img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"error in loading image:{image_path}")
        return img

    def __call__(self, image_path):
        img = self.load_image(image_path)
        dt_boxes, elapse = self.text_detector(img)
        print("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = self.sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            print("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        print("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def main():
    from ch_ppocr_mobile_v2_cls import TextClassifier
    from ch_ppocr_mobile_v2_det import TextDetector
    from ch_ppocr_mobile_v2_rec import TextRecognizer

    det_model_path = 'models/ch_ppocr_mobile_v2.0_det_infer.onnx'
    cls_model_path = 'models/ch_ppocr_mobile_v2.0_cls_infer.onnx'
    rec_model_path = 'models/ch_ppocr_mobile_v2.0_rec_infer.onnx'
    image_path = r'test_images/det_images/1.jpg'

    text_sys = TextSystem(det_model_path,
                          rec_model_path,
                          use_angle_cls=True,
                          cls_model_path=cls_model_path)
    dt_boxes, rec_res = text_sys(image_path)
    visualize(image_path, dt_boxes, rec_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_server', type=str2bool, default=False)

    parser.add_argument('--det_model_path', type=str,
                        default='models/ch_ppocr_mobile_v2.0_det_infer.onnx')
    parser.add_argument('--cls_model_path', type=str,
                        default='models/ch_ppocr_mobile_v2.0_cls_infer.onnx')
    parser.add_argument('--rec_model_path', type=str,
                        default='models/ch_ppocr_mobile_v2.0_rec_infer.onnx')

    parser.add_argument('--image_path', type=str,
                        default='test_images/det_images/1.jpg')
    args = parser.parse_args()

    if args.use_server:
        from ch_ppocr_mobile_v2_cls import TextClassifier
        from ch_ppocr_server_v2_det import TextDetector
        from ch_ppocr_server_v2_rec import TextRecognizer

        if not args.det_model_path.__contains__('server'):
            raise ValueError(f'det模型{args.det_model_path}不是通用模型！')

        if not args.rec_model_path.__contains__('server'):
            raise ValueError(f'rec模型{args.rec_model_path}不是通用模型！')
    else:
        # v2.0 超轻量
        from ch_ppocr_mobile_v2_cls import TextClassifier
        from ch_ppocr_mobile_v2_det import TextDetector
        from ch_ppocr_mobile_v2_rec import TextRecognizer

    if args.det_model_path is None:
        raise FileNotFoundError(f'{args.det_model_path} is not found!')

    if args.cls_model_path is None:
        raise FileNotFoundError(f'{args.cls_model_path} is not found!')

    if args.rec_model_path is None:
        raise FileNotFoundError(f'{args.rec_model_path} is not found!')

    text_sys = TextSystem(args.det_model_path,
                          args.rec_model_path,
                          use_angle_cls=True,
                          cls_model_path=args.cls_model_path)
    dt_boxes, rec_res = text_sys(args.image_path)
    visualize(args.image_path, dt_boxes, rec_res)
