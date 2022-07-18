# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import base64

import cv2
import numpy as np
from flask import Flask, render_template, request
from wsgiref.simple_server import make_server

from task import detect_recognize

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ocr', methods=['POST'])
def ocr():
    if request.method == 'POST':
        url_get = request.get_json()
        url_get = str(url_get).split(',')[1]

        image = base64.b64decode(url_get)
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return detect_recognize(image)


if __name__ == '__main__':
    ip = '0.0.0.0'
    ip_port = 9003

    server = make_server(ip, ip_port, app)
    server.serve_forever()
