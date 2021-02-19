# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: main.py
# @Author: Max
import base64
import json
import time

import cv2
import numpy as np
from flask import Flask, render_template, request

from resources.bpocr import TextSystem, draw_text_det_res

# 实例化模型
det_model_path = 'resources/models/ch_ppocr_server_v2.0_det_infer.onnx'
cls_model_path = 'resources/models/ch_ppocr_mobile_v2.0_cls_infer.onnx'
rec_model_path = 'resources/models/ch_ppocr_mobile_v2.0_rec_infer.onnx'

text_sys = TextSystem(det_model_path,
                      rec_model_path,
                      use_angle_cls=True,
                      cls_model_path=cls_model_path)

app = Flask(__name__)

# 设置上传文件大小
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10M


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ocr', methods=['POST', 'GET'])
def ocr():
    if request.method == 'POST':
        url_get = request.get_json()
        url_get = str(url_get).split(',')[1]
        image = base64.b64decode(url_get)
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
    app.run(host='0.0.0.0', port=9003, debug=False, processes=True)
