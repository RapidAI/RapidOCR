## rapidocr-web
<p>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pypi.org/project/rapidocr-web/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr-web"></a>
</p>

### 1. Install package by pypi.
```bash
$ pip install rapidocr-web
```

### 2. Run by command line.
- Usage:
    ```bash
    $ rapidocr_web -h
    usage: rapidocr_web [-h] [-ip IP] [-p PORT] [-api]

    optional arguments:
    -h, --help            show this help message and exit
    -ip IP, --ip IP       IP Address
    -p PORT, --port PORT  IP port
    -api, --is_api        Whether to use the api format.
    ```
- Example:
    ```bash
    # Web mode
    $ rapidocr_web -ip "0.0.0.0" -p 9003

    # API mode
    $ rapidocr_web -ip "0.0.0.0" -p 9003 -api
    ```

### 3. Use.
- Web mode: Open `http://localhost:9003/` to view, enjoy it.
- API mode:
    ```python
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
        url = 'http://localhost:9003/ocr'
        header = {'Content-Type': 'application/json; charset=UTF-8'}

        img_path = '1.jpg'
        img_json = get_json_format(img_path)

        response = requests.post(url, data=img_json, headers=header)
        if response.status_code == 200:
            rec_res = ast.literal_eval(response.text)
            print(rec_res)
        else:
            print(response.status_code)
    ```

### See details for [RapidOCR](https://github.com/RapidAI/RapidOCR/tree/main/ocrweb).