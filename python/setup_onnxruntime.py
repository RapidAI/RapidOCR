# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import subprocess
from pathlib import Path

import setuptools


def get_latest_version(package_name):
    output = subprocess.run(["pip", "index", "versions", package_name],
                            capture_output=True)
    output = output.stdout.decode('utf-8')
    if output:
        output = list(filter(lambda x: len(x) > 0, output.split('\n')))
        return output[0].strip().split(' ')[-1][1:-1]
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
    readme_path = str(root_dir / 'docs' / 'doc_whl_rapidocr_ort.md')
    print(readme_path)
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


MODULE_NAME = 'rapidocr_onnxruntime'
latest_version = get_latest_version(MODULE_NAME)
VERSION_NUM = version_add_one(latest_version)

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms="Any",
    description="RapidOCR",
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    author="SWHL",
    author_email="liekkaskono@163.com",
    url="https://github.com/RapidAI/RapidOCR",
    license='Apache-2.0',
    include_package_data=True,
    install_requires=["pyclipper>=1.2.1", "onnxruntime>=1.7.0",
                      "opencv_python>=4.5.1.48", "numpy>=1.19.3",
                      "six>=1.15.0", "Shapely>=1.7.1", 'PyYAML'],
    package_dir={'': MODULE_NAME},
    packages=setuptools.find_namespace_packages(where=MODULE_NAME),
    package_data={'': ['*.onnx', '*.yaml']},
    keywords=[
        'ocr,text_detection,text_recognition,db,onnxruntime,paddleocr,openvino,rapidocr'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ]
)
