# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import base64
import copy
import json
from functools import reduce
from pathlib import Path

import cv2
import numpy as np

from rapidocr_onnxruntime import RapidOCR
from detection import detection

text_sys = RapidOCR('config.yaml')


def detect_recognize(image_path, is_api=False):
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        image = image_path
    else:
        raise TypeError(f'{image_path} is not str or ndarray.')

    final_result, img, elapse_part = text_sys(image)

    if is_api:
        final_reuslt_json = json.dumps(final_result,
                                       indent=2,
                                       ensure_ascii=False)
        return final_reuslt_json

    if final_result is None:
        elapse, elapse_part = 0, ''
        img_str = img_to_base64(img)
        rec_res_data = json.dumps([], indent=2, ensure_ascii=False)
    else:
        boxes, txts, scores = list(zip(*final_result))
        # 检测文本扫描结果是否含有恶意代码
        # Provided by BUPT
        txts_str = ''.join(txts)
        if detection(txts_str):
            rec_res_data = json.dumps('1', indent=2, ensure_ascii=False)
        else:
            rec_res = list(zip(range(len(txts)), txts, scores))
            rec_res_data = json.dumps(rec_res,
                                      indent=2,
                                      ensure_ascii=False)

        det_im = draw_text_det_res(np.array(boxes), img)
        img_str = img_to_base64(det_im)

        elapse = reduce(lambda x, y: float(x)+float(y), elapse_part)
        elapse_part = ','.join([str(x) for x in elapse_part])
    return img_str, elapse, elapse_part, rec_res_data


def img_to_base64(img):
    img = cv2.imencode('.jpg', img)[1]
    img_str = str(base64.b64encode(img))[2:-1]
    return img_str


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

def check_image_type(file):
    '''
    验证image是不是一张真图片
    Provided by BUPT
    '''

    # check whether the file is in image format
    is_pic_type = isinstance(file, np.ndarray)
    if is_pic_type:
        height, width = file.shape[:2]
        # Use the height and width attributes to verify whether the image is a true image
        if height > 0 and width > 0:
            print("上传文件格式正确 ")
            return True
        else:
            print("请选择正确的图片格式")
            return False

    else:
        print("请选择正确的图片格式")
        return False