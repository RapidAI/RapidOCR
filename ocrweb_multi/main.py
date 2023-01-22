# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import logging
import cv2
import numpy as np
from flask import Flask, send_file, request, make_response
from waitress import serve


from rapidocr.main import detect_recognize
from utils.config import conf
from utils.utils import tojson, parse_bool

app = Flask(__name__)
log = logging.getLogger('app')
# 设置上传文件大小
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

@app.route('/')
def index():
    return send_file('static/index.html')


def json_response(data, status=200):
    return make_response(tojson(data), status, {"content-type": 'application/json'})


@app.route('/lang')
def get_languages():
    """返回可用语言列表"""
    data = [
        {'code': key, 'name': val['name']} for key, val in conf['languages'].items()
    ]
    result = {'msg': 'OK', 'data': data}
    log.info('Send langs: %s', data)
    return json_response(result)


@app.route('/ocr', methods=['POST', 'GET'])
def ocr():
    """执行文字识别"""
    if conf['server'].get('token'):
        if request.values.get('token') != conf['server']['token']:
            return json_response({'msg': 'invalid token'}, status=403)

    lang = request.values.get('lang') or 'ch'
    detect = parse_bool(request.values.get('detect') or 'true')
    classify = parse_bool(request.values.get('classify') or 'true')

    image_file = request.files.get('image')
    if not image_file:
        return json_response({'msg': 'no image'}, 400)
    nparr = np.frombuffer(image_file.stream.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    log.info('Input: image %s, lang=%s, detect=%s, classify=%s', image.shape, lang, detect, classify)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result = detect_recognize(image, lang=lang, detect=detect, classify=classify)
    log.info('OCR Done %s %s', result['ts'], len(result['results']))
    return json_response({'msg': 'OK', 'data': result})


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    logging.getLogger('waitress').setLevel(logging.INFO)
    if parse_bool(conf.get('debug', '0')):
        # Debug
        app.run(host=conf['server']['host'], port=conf['server']['port'], debug=True)
    else:
        # Deploy with waitress
        serve(app, host=conf['server']['host'], port=conf['server']['port'])
