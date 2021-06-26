# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: main.py
# @Author: Max
import base64

import cv2
import numpy as np
from flask import Flask, render_template, request

from task import detect_recognize

app = Flask(__name__)

# 设置上传文件大小
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024


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
        return detect_recognize(image)


if __name__ == '__main__':
    app.run(host='127.0.0.1',
            port=9003,
            debug=False,
            processes=True)
