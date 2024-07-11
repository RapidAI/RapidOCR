# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import base64
import io
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
        self.ocr = RapidOCR()

    def __call__(self, img: Image.Image) -> Dict:
        img = np.array(img)
        ocr_res, _ = self.ocr(img)

        if not ocr_res:
            return {}

        out_dict = {
            str(i): {"rec_txt": rec, "dt_boxes": dt_box, "score": score}
            for i, (dt_box, rec, score) in enumerate(ocr_res)
        }

        return self.convert_to_json(out_dict)

    def convert_to_json(self, data):
        """Recursively convert NumPy types to Python types."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.generic):
            return data.item()
        elif isinstance(data, (list, tuple)):
            return [self.convert_to_json(item) for item in data]
        elif isinstance(data, dict):
            return {k: self.convert_to_json(v) for k, v in data.items()}
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data

app = FastAPI()
processor = OCRAPIUtils()


@app.get("/")
async def root():
    return {"message": "Welcome to RapidOCR Server!"}


@app.post("/ocr")
async def ocr(image_file: UploadFile = None, image_data: str = Form(None)):
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

    ocr_res = processor(img)
    return ocr_res


def main():
    parser = argparse.ArgumentParser("rapidocr_api")
    parser.add_argument("-ip", "--ip", type=str, default="0.0.0.0", help="IP Address")
    parser.add_argument("-p", "--port", type=int, default=9003, help="IP port")
    args = parser.parse_args()

    cur_file_path = Path(__file__).resolve()
    app_path = f"{cur_file_path.parent.name}.{cur_file_path.stem}:app"
    print(app_path)
    uvicorn.run(app_path, host=args.ip, port=args.port, reload=True)


if __name__ == "__main__":
    main()
