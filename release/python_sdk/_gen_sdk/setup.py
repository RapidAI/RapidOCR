# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: setup.py


import setuptools

setuptools.setup(
    name="rapidocr",
    version="1.0.0",
    platforms="Windows & Linux & Mac",
    description="RapidOCR",
    author="SWHL",
    author_email="liekkaskono@163.com",
    url="https://github.com/RapidAI/RapidOCR",
    license='Apache-2.0',
    py_modules="rapidocr",
    include_package_data=True,
    install_requires=["pyclipper>=1.2.1", "onnxruntime",
                      "opencv_python>=4.5.1.48", "numpy>=1.19.3",
                      "six>=1.15.0", "Shapely>=1.7.1"],
    packages=setuptools.find_packages()
)
