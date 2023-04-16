# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import json

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


class OCRAPIUtils():
    def __init__(self) -> None:
        self.ocr = RapidOCR()

    def __call__(self, img):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        ocr_res, _ = self.ocr(img)
        if not ocr_res:
            return json.dumps({})

        out_dict = {str(i): {'rec_txt': rec,
                             'dt_boxes': dt_box,
                             'score': score}
                    for i, (dt_box, rec, score) in enumerate(ocr_res)}
        return json.dumps(out_dict, indent=2, ensure_ascii=False)


app = FastAPI()
processor = OCRAPIUtils()


@app.get("/")
async def root():
    return {'message': 'Welcome to RapidOCR Server!'}


@app.post('/ocr')
async def ocr(image: UploadFile):
    image = Image.open(image.file)
    ocr_res = processor(image)
    return ocr_res


def main():
    parser = argparse.ArgumentParser('rapidocr_api')
    parser.add_argument('-ip', '--ip', type=str, default='0.0.0.0',
                        help='IP Address')
    parser.add_argument('-p', '--port', type=int, default=9003,
                        help='IP port')
    args = parser.parse_args()
    uvicorn.run(app, host=args.ip, port=args.port)


if __name__ == '__main__':
    main()
