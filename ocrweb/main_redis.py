# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: main.py
# @Author: Max
import base64
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
import redis
from flask import Flask, render_template, request
from rq import Queue
from rq.job import Job


from task_redis import conn, detect_recognize


root_path = Path(__file__).resolve().parent


r = redis.Redis(host='localhost',
                port=6379,
                decode_responses=True)

# 建立与Redis server的连接并初始化一个队列
q = Queue(connection=conn, default_timeout=5000)

app = Flask(__name__)

# 设置上传文件大小
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10M


@app.route('/')
def index():
    return render_template('index_redis.html')


@app.route('/ocr', methods=['POST', 'GET'])
def ocr():
    if request.method == 'POST':
        # 获取图像数据
        url_get = request.get_json()
        url_get = str(url_get).split(',')[1]
        image = base64.b64decode(url_get)
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        img_name = str(uuid.uuid4())

        image_directory = root_path / 'static' / 'images'
        if not image_directory.exists():
            image_directory.mkdir(parents=True, exist_ok=True)

        img_path = str(image_directory / f'{img_name}.jpg')
        cv2.imwrite(img_path, image)

        # 压入队列
        job = q.enqueue(detect_recognize,
                        args=(img_path, ),
                        result_ttl=5000)
        return json.dumps({'job_id': job.get_id()})


@app.route('/get_results', methods=['POST'])
def get_results():
    job_id = request.get_json()
    job = Job.fetch(job_id['job_id'], connection=conn)
    if job.is_finished:
        job_result = eval(job.result)
        job_result['status'] = True
        return json.dumps(job_result)
    else:
        return json.dumps({'status': False})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9003,
            debug=False, processes=True,
            threaded=True)
