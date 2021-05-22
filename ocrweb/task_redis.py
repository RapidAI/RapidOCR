# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: worker.py
# @Time: 2021/03/07 20:29:32
# @Author: Max
import json
import base64
import cv2
import time
import numpy as np

import redis
from rq import Worker, Queue, Connection
from resources.bpocr import TextSystem, draw_text_det_res

listen = ['default']
redis_url = "redis://localhost:6379"  # redis server 默认地址
conn = redis.from_url(redis_url)

# 实例化模型
det_model_path = 'resources/models/ch_ppocr_server_v2.0_det_infer.onnx'
cls_model_path = 'resources/models/ch_ppocr_mobile_v2.0_cls_infer.onnx'
rec_model_path = 'resources/models/ch_ppocr_mobile_v2.0_rec_infer.onnx'

text_sys = TextSystem(det_model_path,
                      rec_model_path,
                      use_angle_cls=True,
                      cls_model_path=cls_model_path)


def detect_recognize(image_path):
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        image = image_path

    t1 = time.time()
    dt_boxes, rec_res, img = text_sys(image)
    t2 = time.time()

    if dt_boxes is None or rec_res is None:
        temp_rec_res = []
        rec_res_data = json.dumps(temp_rec_res,
                                  indent=2,
                                  ensure_ascii=False)
        elapse = 0
        image = cv2.imencode('.jpg', img)[1]
        img = str(base64.b64encode(image))[2:-1]
    else:
        temp_rec_res = []
        for i, value in enumerate(rec_res):
            temp_rec_res.append([i, value[0], value[1]])
        temp_rec_res = np.array(temp_rec_res)
        rec_res_data = json.dumps(temp_rec_res.tolist(),
                                  indent=2,
                                  ensure_ascii=False)

        det_im = draw_text_det_res(dt_boxes, img)
        image = cv2.imencode('.jpg', det_im)[1]
        img = str(base64.b64encode(image))[2:-1]

        elapse = f'{t2-t1:.3f}'
    return json.dumps({'image': img,
                       'elapse': elapse,
                       'rec_res': rec_res_data})


if __name__ == '__main__':
    with Connection(conn):  # 建立与redis server的连接
        worker = Worker(list(map(Queue, listen)))  # 建立worker监听给定的队列
        worker.work()