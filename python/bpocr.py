# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import copy
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ch_ppocr_mobile_v1_det import TextDetector
from ch_ppocr_mobile_v2_rec import TextRecognizer

font_path = r'resources\simfang.ttf'
drop_score = 0.5


def test_text_det():
    # 单独跑文本检测模型
    det_model_path = 'models\ch_mobile_v1.1_det.onnx'
    image_path = r'test_images\det_images\1.jpg'

    test_detector = TextDetector(det_model_path)

    # dst_boxes: 检测到图像中的文本框坐标，ndarray格式
    # (10, 4, 2)→[10个，4个坐标，每个坐标两个点]
    dt_boxes, elapse, ori_im = test_detector(image_path)

    cv2.imwrite('det_results.jpg', im)
    print('图像已经保存在了det_results.jpg中')


def test_text_rec():
    # 单独跑文本识别模型
    rec_model_path = r'models\ch_ppocr_mobile_v2.0_rec_pre_infer.onnx'
    image_path = r'test_images\rec_images\2021-01-19_13-44-34.png'
    text_recongnizer = TextRecognizer(rec_model_path)

    rec_res, elapse = text_recongnizer(image_path)
    print(f'识别结果：{rec_res}\tcost: {elapse}s')


def draw_ocr_box_txt(image, boxes, txts, scores=None, drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

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


def visualize(img, boxes, rec_res, image_file):
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    txts = [rec_res[i][0] for i in range(len(rec_res))]
    scores = [rec_res[i][1] for i in range(len(rec_res))]

    draw_img = draw_ocr_box_txt(image, boxes, txts, scores,
                                drop_score=drop_score,
                                font_path=font_path)
    draw_img_save = "./inference_results/"
    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    cv2.imwrite(
        os.path.join(draw_img_save, os.path.basename(image_file)),
        draw_img[:, :, ::-1])


class TextSystem(object):
    def __init__(self, det_model_path, rec_model_path) -> None:
        super(TextSystem).__init__()
        self.text_detector = TextDetector(det_model_path)
        self.text_recognizer = TextRecognizer(rec_model_path)
        self.use_angle_cls = False

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

    def __call__(self, img_path):
        dt_boxes, elapse, ori_im = self.text_detector(img_path)
        print("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = self.sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
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


if __name__ == '__main__':
    # test_text_det()
    # test_text_rec()

    # 联合测试
    det_model_path = 'models\ch_mobile_v1.1_det.onnx'
    rec_model_path = 'models\ch_ppocr_mobile_v2.0_rec_pre_infer.onnx'
    image_path = r'test_images\1.jpg'
    text_sys = TextSystem(det_model_path, rec_model_path)
    dt_boxes, rec_res = text_sys(image_path)
    print(dt_boxes)
    print(rec_res)
    img = cv2.imread(image_path)
    visualize(img, dt_boxes, rec_res, image_path)
