# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com

# import requests

# url = "http://localhost:9003/ocr"
# img_path = "../python/tests/test_files/ch_en_num.jpg"

# with open(img_path, "rb") as f:
#     file_dict = {"image_file": (img_path, f, "image/png")}
#     response = requests.post(url, files=file_dict, timeout=60)

# print(response.text)

import base64

import requests

url = "http://localhost:9003/ocr"
img_path = "../python/tests/test_files/ch_en_num.jpg"

with open(img_path, "rb") as fa:
    img_str = base64.b64encode(fa.read())

payload = {"image_data": img_str}
response = requests.post(url, data=payload, timeout=60)

print(response.json())
