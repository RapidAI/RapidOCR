# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
from pathlib import Path
from wsgiref.simple_server import make_server

from flask import Flask, render_template, request
try:
    from .task import OCRWebUtils
except:
    from task import OCRWebUtils

root_dir = Path(__file__).resolve().parent

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024
processor = OCRWebUtils()


@app.route('/')
def index():
    if not app.config['is_api']:
        return render_template('index.html')


@app.route('/ocr', methods=['POST'])
def ocr():
    if request.method == 'POST':
        img_str = request.get_json().get('file', None)
        ocr_res = processor(img_str, is_api=app.config['is_api'])
        return ocr_res


def main():
    parser = argparse.ArgumentParser('rapidocr_web')
    parser.add_argument('-ip', '--ip', type=str, default='0.0.0.0',
                        help='IP Address')
    parser.add_argument('-p', '--port', type=int, default=9003,
                        help='IP port')
    parser.add_argument('-api', '--is_api', action='store_true',
                        default=False, help='Whether to use the api format.')
    args = parser.parse_args()

    app.config['is_api'] = args.is_api
    server = make_server(args.ip, args.port, app)
    server.serve_forever()


if __name__ == '__main__':
    main()
