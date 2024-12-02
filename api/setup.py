# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path
from typing import List

import setuptools
from get_pypi_latest_version import GetPyPiLatestVersion


def read_txt(txt_path: str) -> List:
    if not isinstance(txt_path, str):
        txt_path = str(txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        data = list(map(lambda x: x.rstrip("\n"), f))
    return data


def get_readme():
    root_dir = Path(__file__).resolve().parent.parent
    readme_path = str(root_dir / "docs" / "doc_whl_rapidocr_api.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    return readme


MODULE_NAME = "rapidocr_api"

obtainer = GetPyPiLatestVersion()
try:
    latest_version = obtainer(MODULE_NAME)
except ValueError:
    latest_version = "0.0.1"
VERSION_NUM = obtainer.version_add_one(latest_version)

if len(sys.argv) > 2:
    match_str = " ".join(sys.argv[2:])
    matched_versions = obtainer.extract_version(match_str)
    if matched_versions:
        VERSION_NUM = matched_versions
sys.argv = sys.argv[:2]

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms="Any",
    description="A cross platform OCR API Library based on OnnxRuntime.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    author="SWHL",
    author_email="liekkaskono@163.com",
    url="https://github.com/RapidAI/RapidOCR",
    download_url="https://github.com/RapidAI/RapidOCR.git",
    license="Apache-2.0",
    include_package_data=True,
    install_requires=read_txt("requirements.txt"),
    extras_require={
        'onnx': ['rapidocr-onnxruntime'],
        'paddle': ['rapidocr-paddle'],
        'openvino': ['rapidocr-openvino'],
    },
    packages=[MODULE_NAME],
    package_data={"": ["*.ico", "*.css", "*.js", "*.html"]},
    keywords=[
        "ocr,text_detection,text_recognition,db,onnxruntime,paddleocr,openvino,rapidocr"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.6,<3.13",
    entry_points={
        "console_scripts": [
            f"{MODULE_NAME}={MODULE_NAME}.main:main",
        ],
    },
)
