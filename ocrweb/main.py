# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import base64
import json
from wsgiref.simple_server import make_server

import cv2
import numpy as np
from flask import Flask, render_template, request

from task import detect_recognize, check_pic_type

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ocr', methods=['POST'])
def ocr():
    if request.method == 'POST':
        url_get = request.get_json()
        img_str = str(url_get).split(',')[1]

        image = base64.b64decode(img_str + '=' * (-len(img_str) % 4))
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 检查后缀名是图片格式，但实际不是图片的文件
        # Provided by BUPT
        is_pic = check_pic_type(image)
        if is_pic:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            img, elapse, elapse_part, rec_res_data, det_str = detect_recognize(image)
            return json.dumps({'image': img,
                               'total_elapse': f'{elapse:.4f}',
                               'elapse_part': elapse_part,
                               'rec_res': rec_res_data,
                               'det_str': det_str})
        else:
            return json.dumps({'message': '请选择正确的图片格式'})


if __name__ == '__main__':
    ip = '0.0.0.0'
    ip_port = 9003

    server = make_server(ip, ip_port, app)
    server.serve_forever()