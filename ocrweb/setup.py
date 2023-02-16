# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import re
import subprocess
import sys
from pathlib import Path

import setuptools


def get_latest_version(package_name):
    output = subprocess.run(["pip", "index", "versions", package_name],
                            capture_output=True)
    output = output.stdout.decode('utf-8')
    if output:
        return extract_version(output)
    return None


def version_add_one(version, add_loc=-1):
    if version:
        version_list = version.split('.')
        mini_version = str(int(version_list[add_loc]) + 1)
        version_list[add_loc] = mini_version
        new_version = '.'.join(version_list)
        return new_version
    return '0.0.0'


def get_readme():
    root_dir = Path(__file__).resolve().parent.parent
    readme_path = str(root_dir / 'docs' / 'doc_whl_rapidocr_web.md')
    print(readme_path)
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


def extract_version(message: str) -> str:
    pattern = r'\d+\.(?:\d+\.)*\d+'
    matched_versions = re.findall(pattern, message)
    if matched_versions:
        return matched_versions[0]
    return ''


MODULE_NAME = 'rapidocr_web'
latest_version = get_latest_version(MODULE_NAME)
VERSION_NUM = version_add_one(latest_version)

# 优先提取commit message中的语义化版本号，如无，则自动加1
if len(sys.argv) > 2:
    match_str = ' '.join(sys.argv[2:])
    matched_versions = extract_version(match_str)
    if matched_versions:
        VERSION_NUM = matched_versions
sys.argv = sys.argv[:2]

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms="Any",
    description="A cross platform OCR Library based on OnnxRuntime.",
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    author="SWHL",
    author_email="liekkaskono@163.com",
    url="https://github.com/RapidAI/RapidOCR",
    download_url='https://github.com/RapidAI/RapidOCR.git',
    license='Apache-2.0',
    include_package_data=True,
    install_requires=["Pillow>=1.2.1", "requests",
                      "Flask>=2.1.0", 'rapidocr_onnxruntime'],
    packages=[MODULE_NAME,
              f'{MODULE_NAME}.static.css',
              f'{MODULE_NAME}.static.js',
              f'{MODULE_NAME}.templates'],
    package_data={'': ['*.ico', '*.css', '*.js', '*.html']},
    keywords=[
        'ocr,text_detection,text_recognition,db,onnxruntime,paddleocr,openvino,rapidocr'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [f'{MODULE_NAME}={MODULE_NAME}.ocrweb:main'],
    }
)
