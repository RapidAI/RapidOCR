# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: setup.py
# @Time: 2021/05/07 16:47:05
# @Author: Max

import setuptools

setuptools.setup(
    name="rapidocr",
    version="1.0.0",
    platforms="Windows & Linux",
    description="RapidOCR",
    author="SWHL",
    author_email="liekkaskono@163.com",
    url="https://github.com/RapidAI/RapidOCR",
    license='Apache-2.0',
    py_modules="rapidocr",
    include_package_data=True,
    install_requires=["pyclipper==1.2.1", "onnxruntime==1.7.0",
                      "opencv_python==4.5.1.48", "numpy==1.19.5",
                      "six==1.15.0", "Shapely>=1.7.1"],
    packages=setuptools.find_packages()
)
