import ast
import base64
import json

import requests


def get_json_format(img_path):
    with open(img_path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_json = json.dumps({'file': img_byte.decode('ascii')})
    return img_json


if __name__ == '__main__':
    url = 'http://localhost:5000/ocr'
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTY3NjgyNDUyMiwianRpIjoiMmZlMjU4OTktYzM2MS00ZjBlLTg0ZWItMDNmYzFkMGI1YmYwIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6InRlc3QiLCJuYmYiOjE2NzY4MjQ1MjIsImV4cCI6MTY3NjgyNDU4Mn0.hMZjPEBb5t0AIm0gHZQ3lwsfEDzdjQiMLo-FVYnepTQ'
    header = {'Content-Type': 'application/json; charset=UTF-8',
		'Authorization':"Bearer "+ token}

    #img_path = '1.png'
    #img_json = get_json_format(img_path)

    response = requests.post(url, headers=header)
    if response.status_code == 200:
        #rec_res = ast.literal_eval(response.text)
        print(response.text)
    else:
        print(response.status_code)
	#print(response.text)
