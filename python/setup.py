# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import setuptools


with open('README.md', 'r') as f:
  README = f.read()

module_name = 'rapidocr_onnxruntime'

setuptools.setup(
    name=module_name,
    version="1.0.0",
    platforms="Windows & Linux & Mac",
    description="RapidOCR",
    long_description=README,
    long_description_content_type='text/markdown',
    author="SWHL",
    author_email="liekkaskono@163.com",
    url="https://github.com/RapidAI/RapidOCR",
    license='Apache-2.0',
    include_package_data=True,
    install_requires=["pyclipper>=1.2.1", "onnxruntime",
                      "opencv_python>=4.5.1.48", "numpy>=1.19.3",
                      "six>=1.15.0", "Shapely>=1.7.1"],
    package_dir={'': module_name},
    packages=setuptools.find_packages(where=module_name)
)

