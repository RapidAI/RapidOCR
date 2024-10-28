# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, Form, UploadFile
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

sys.path.append(str(Path(__file__).resolve().parent.parent))


class OCRAPIUtils:
    def __init__(self) -> None:
        # 从环境变量中读取参数
        det_model_path = os.getenv("det_model_path", None)
        cls_model_path = os.getenv("cls_model_path", None)
        rec_model_path = os.getenv("rec_model_path", None)

        self.ocr = RapidOCR(det_model_path=det_model_path, cls_model_path=cls_model_path, rec_model_path=rec_model_path)

    def __call__(self, img: Image.Image, use_det=None, use_cls=None, use_rec=None) -> Dict:
        img = np.array(img)
        ocr_res, _ = self.ocr(img, use_det=use_det, use_cls=use_cls, use_rec=use_rec)

        if not ocr_res:
            return {}

        # 转换为字典格式: 兼容所有参数情况
        out_dict = {}
        for i, dats in enumerate(ocr_res):
            values = {}
            for dat in dats:
                if type(dat) == str:
                    values["rec_txt"] = dat
                if type(dat) == np.float64:
                    values["score"] = f"{dat:.4f}"
                if type(dat) == list:
                    values["dt_boxes"] = dat
            out_dict[str(i)] = values

        return out_dict


app = FastAPI()
processor = OCRAPIUtils()

@app.get("/")
async def root():
    return {"message": "Welcome to RapidOCR API Server!"}

@app.post("/ocr")
async def ocr(image_file: UploadFile = None, image_data: str = Form(None),
        use_det: bool = Form(None), use_cls: bool = Form(None), use_rec: bool = Form(None)):

    if image_file:
        img = Image.open(image_file.file)
    elif image_data:
        img_bytes = str.encode(image_data)
        img_b64decode = base64.b64decode(img_bytes)
        img = Image.open(io.BytesIO(img_b64decode))
    else:
        raise ValueError(
            "When sending a post request, data or files must have a value."
        )
    ocr_res = processor(img, use_det=use_det, use_cls=use_cls, use_rec=use_rec)
    return ocr_res

def main():
    parser = argparse.ArgumentParser("rapidocr_api")
    parser.add_argument("-ip", "--ip", type=str, default="0.0.0.0", help="IP Address")
    parser.add_argument("-p", "--port", type=int, default=9003, help="IP port")
    parser.add_argument('-workers', "--workers", type=int, default=1, help='number of worker process')
    args = parser.parse_args()

    uvicorn.run("rapidocr_api.main:app", host=args.ip, port=args.port, reload=0, workers=args.workers)

if __name__ == "__main__":
    main()
