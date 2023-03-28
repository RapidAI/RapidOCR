# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import setuptools
from get_pypi_latest_version import GetPyPiLatestVersion


def get_readme():
    root_dir = Path(__file__).resolve().parent.parent
    readme_path = str(root_dir / 'docs' / 'doc_whl_rapidocr_web.md')
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


MODULE_NAME = 'rapidocr_web'

obtainer = GetPyPiLatestVersion()
latest_version = obtainer(MODULE_NAME)
VERSION_NUM = obtainer.version_add_one(latest_version)

# 优先提取commit message中的语义化版本号，如无，则自动加1
if len(sys.argv) > 2:
    match_str = ' '.join(sys.argv[2:])
    matched_versions = obtainer.extract_version(match_str)
    if matched_versions:
        VERSION_NUM = matched_versions
sys.argv = sys.argv[:2]

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms='Any',
    description='A cross platform OCR Library based on OnnxRuntime.',
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    author='SWHL',
    author_email='liekkaskono@163.com',
    url='https://github.com/RapidAI/RapidOCR',
    download_url='https://github.com/RapidAI/RapidOCR.git',
    license='Apache-2.0',
    include_package_data=True,
    install_requires=['requests', 'Flask>=2.1.0', 'rapidocr_onnxruntime'],
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
    python_requires='>=3.6,<=3.10',
    entry_points={
        'console_scripts': [f'{MODULE_NAME}={MODULE_NAME}.ocrweb:main'],
    },
)
