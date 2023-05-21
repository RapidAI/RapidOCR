## rapidocr-web
<p>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href="https://pypi.org/project/rapidocr-web/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr-web"></a>
    <a href="https://pepy.tech/project/rapidocr_web"><img src="https://static.pepy.tech/personalized-badge/rapidocr_web?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
</p>

### use
1. Install package by pypi.
    ```bash
    $ pip install rapidocr-web
    ```
2. Run by command line.
   - Usage:
       ```bash
       $ rapidocr_web -h
       usage: rapidocr_web [-h] [-ip IP] [-p PORT]

       optional arguments:
       -h, --help            show this help message and exit
       -ip IP, --ip IP       IP Address
       -p PORT, --port PORT  IP port
       ```
   - Example:
       ```bash
       $ rapidocr_web -ip "0.0.0.0" -p 9003
       ```
3. Open `http://localhost:9003/` to view, enjoy it.

### See details for [RapidOCR](https://github.com/RapidAI/RapidOCR/tree/main/ocrweb).