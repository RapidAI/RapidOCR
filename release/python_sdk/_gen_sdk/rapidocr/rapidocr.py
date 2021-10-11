# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import copy
import imghdr
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .text_cls import TextDirectionClassifier
    from .text_detect import TextDetector
    from .text_recognize import TextRecognizer
except:
    from text_cls import TextDirectionClassifier
    from text_detect import TextDetector
    from text_recognize import TextRecognizer

drop_score = 0.5


def get_image_file_list(img_file):
    if img_file is None or not Path(img_file).exists():
        raise FileExistsError(f"not found any img file in {img_file}")

    imgs_lists = []
    img_end = ['jpg', 'JPG', 'bmp', 'BMP', 'png', 'PNG', 'jpeg', 'JPEG',
               'rgb', 'tif', 'tiff', 'gif', 'GIF']

    def _get_img_path(img_file):
        # 真实后缀
        real_img_suffix = imghdr.what(img_file)

        # 现在后缀
        now_img_suffix = img_file.suffix[1:]
        if real_img_suffix in img_end:
            # 真实后缀与现有后缀不一致
            if now_img_suffix in img_end:
                return img_file
            else:
                # 按照真实后缀重命名
                new_img_file = img_file.with_suffix(f'.{real_img_suffix}')
                img_file.rename(new_img_file)
                return new_img_file
        else:
            return None

    img_file = Path(img_file)
    if img_file.is_file():
        tmp_img_path = _get_img_path(img_file)
        if tmp_img_path is not None:
            imgs_lists.append(str(tmp_img_path))

    elif img_file.is_dir():
        for img_path in img_file.iterdir():
            tmp_file_path = _get_img_path(img_path)
            if tmp_file_path is not None:
                imgs_lists.append(str(tmp_file_path))
    else:
        raise ValueError(f"The {img_file}'s format is not support!")

    if len(imgs_lists) == 0:
        raise FileExistsError(f"not found any img file in {img_file}")

    return imgs_lists


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


def draw_ocr_box_txt(image, boxes, txts,
                     scores=None, drop_score=0.5,
                     font_path="../assets/simsun.ttf"):
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
            draw_right.text([box[0][0], box[0][1]],
                            txt,
                            fill=(0, 0, 0),
                            font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def draw_box_txt(image, boxes, txts,
                 scores=None, drop_score=0.5,
                 font_path="assets/simsun.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right_width = int(w * 0.3)
    img_right = Image.new('RGB', (img_right_width, h), (255, 255, 255))

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    cur_y = 0
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2
                               + (box[0][1] - box[3][1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2
                              + (box[0][1] - box[1][1])**2)
        if box_height > 2 * box_width:
            # 竖直文本框
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size,
                                      encoding="utf-8")
            cur_y = 0
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            font_w, font_h = font.getsize(txt)
            draw_right.text([10, cur_y],
                            txt,
                            fill=(0, 0, 0),
                            font=font)
            cur_y += font_h

    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (img_right_width + w, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, img_right_width + w, h))
    return np.array(img_show)


def visualize(image_path, boxes, rec_res):
    image, flag = check_and_read_gif(image_path)
    if not flag:
        image = cv2.imread(image_path)
    txts = [rec_res[i][0] for i in range(len(rec_res))]
    scores = [rec_res[i][1] for i in range(len(rec_res))]

    tmp_img = draw_text_det_res(boxes, np.array(image))
    draw_img = draw_ocr_box_txt(Image.fromarray(tmp_img), boxes,
                                txts, scores,
                                drop_score=0.5)

    img_directory = Path(image_path).parent.name
    draw_img_save = Path("./test_images/inference_results") / img_directory
    if not draw_img_save.exists():
        draw_img_save.mkdir(parents=True,
                            exist_ok=True)
    image_save = str(draw_img_save / f'infer_{Path(image_path).name}.jpg')

    cv2.imwrite(image_save, draw_img[:, :, ::-1])
    for value in txts:
        print(value)
    write_txt(str(draw_img_save / f'{Path(image_path).stem}.txt'), txts)
    print(f'结果保存: {image_save}')


def write_txt(save_path: str, content: list, mode='w'):
    """
    将list内容写入txt中
    @param
    content: list格式内容
    save_path: 绝对路径str
    @return:None
    """
    with open(save_path, mode, encoding='utf-8') as f:
        for value in content:
            f.write(value + '\n')


class TextSystem(object):
    def __init__(self,
                 det_model_path,
                 rec_model_path,
                 direct_cls_model_path,
                 dict_path) -> None:
        super(TextSystem).__init__()

        # 验证模型是否存在
        self.verify_exist(det_model_path)
        self.verify_exist(rec_model_path)
        self.verify_exist(direct_cls_model_path)
        self.verify_exist(dict_path)

        self.text_detector = TextDetector(det_model_path)
        self.text_recognizer = TextRecognizer(rec_model_path, dict_path)
        self.text_classifier = TextDirectionClassifier(direct_cls_model_path)

    @staticmethod
    def verify_exist(model_path):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f'{model_path} does not exist!')

        if not model_path.is_file():
            raise FileExistsError(f'{model_path} must be a file')

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
        points = points.astype(pts_std.dtype)
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

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            raise ValueError("The format of the img is not ndarray")

        dt_boxes, elapse = self.text_detector(img)
        if dt_boxes is None or len(dt_boxes) < 1:
            return None, None
        img_crop_list = []

        # 检测框排序
        dt_boxes = self.sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)

        img_crop_list, _, elapse = self.text_classifier(img_crop_list)

        rec_res, elapse = self.text_recognizer(img_crop_list)

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        return filter_boxes, filter_rec_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--det_model_path', type=str,
                        default='models/general_mobile_v1_det_infer.onnx')

    parser.add_argument('--direct_cls_model_path', type=str,
                        default='models/ch_ppocr_mobile_v2.0_cls_infer.onnx')

    parser.add_argument('--rec_model_path', type=str,
                        default='models/mobile_korean_rec_infer.onnx')

    parser.add_argument('--image_dir', type=str,
                        default='test_images/korean/det_rec/2021-03-01_15-29-50.png')

    args = parser.parse_args()

    if args.det_model_path is None:
        raise FileNotFoundError(f'{args.det_model_path} is not found!')

    if args.direct_cls_model_path is None:
        raise FileNotFoundError(f'{args.direct_cls_model_path} is not found!')

    if args.rec_model_path is None:
        raise FileNotFoundError(f'{args.rec_model_path} is not found!')

    text_sys = TextSystem(args.det_model_path,
                          args.rec_model_path,
                          use_direct_cls=True,
                          direct_cls_model_path=args.direct_cls_model_path)
    img_file_list = get_image_file_list(args.image_dir)
    for image_path in img_file_list:
        try:
            dt_boxes, rec_res = text_sys(image_path)
            if dt_boxes is not None:
                visualize(image_path, dt_boxes, rec_res)
        except ValueError:
            continue
