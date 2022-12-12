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
from detection import Detection

text_sys = RapidOCR('config.yaml')


def detect_recognize(image_path, is_api=False, security_check = False):
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
        det_str = detection(final_result)

        rec_res = list(zip(range(len(txts)), txts, scores))
        rec_res_data = json.dumps(rec_res,
                                  indent=2,
                                  ensure_ascii=False)
        det_im = draw_text_det_res(np.array(boxes), img)
        img_str = img_to_base64(det_im)

        elapse = reduce(lambda x, y: float(x)+float(y), elapse_part)
        elapse_part = ','.join([str(x) for x in elapse_part])
    return img_str, elapse, elapse_part, rec_res_data, det_str


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


def check_pic_type(image):
    '''
    验证image是不是一张真图片（如：后缀名是图片格式，但实际不是图片的文件）
    Provided by BUPT
    '''

    # check whether the file is in image format
    is_pic_type = isinstance(image, np.ndarray)
    if is_pic_type:
        height, width = image.shape[:2]
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


def detection(rec_res_data, js=True, css=True, url=True, il_word=True):
    """
    综合判断OCR过程中是否存在安全问题（包括：js、css、url注入，违规词汇检测）
    Provided by BUPT
    """
    det_str = ''
    det_dict = {'js': False, 'css': False, 'url': False, 'il_word': False}
    det = Detection(rec_res_data)

    if js:
        det_dict['js'] = det.js_test()
    if css:
        det_dict['css'] = det.css_test()
    if url:
        det_dict['url'] = det.url_test()
    if il_word:
        det_dict['il_word'] = det.il_word_test()

    s_str = '疑似存在安全隐患：'
    e_str = '请检查！'
    js_str = 'js注入；'
    css_str = 'css注入；'
    url_str = 'url注入；'
    il_word_str = '违规词汇；'
    for key, value in det_dict.items():
        if key == 'js' and value:
            det_str += js_str
        if key == 'css' and value:
            det_str += css_str
        if key == 'url' and value:
            det_str += url_str
        if key == 'il_word' and value:
            det_str += il_word_str
    if det_str != '':
        det_str = s_str + det_str + e_str
    return det_str