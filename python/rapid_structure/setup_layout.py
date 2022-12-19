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
        return output[0].split(' ')[-1][1:-1]
    return None


def version_add_one(version, add_loc=-1):
    if version:
        version_list = version.split('.')
        mini_version = str(int(version_list[add_loc]) + 1)
        version_list[add_loc] = mini_version
        new_version = '.'.join(version_list)
        return new_version
    return '0.0.1'


def get_readme():
    root_dir = Path(__file__).resolve().parent.parent.parent
    readme_path = str(root_dir / 'docs' / 'doc_whl_rapid_layout.md')
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


MODULE_NAME = 'rapid_layout'
latest_version = get_latest_version(MODULE_NAME)
VERSION_NUM = version_add_one(latest_version)

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms="Any",
    description="RapidLayout",
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    author="SWHL",
    author_email="liekkaskono@163.com",
    url="https://github.com/RapidAI/RapidOCR",
    license='Apache-2.0',
    include_package_data=True,
    install_requires=["onnxruntime>=1.7.0", "PyYAML>=6.0",
                      "opencv_python>=4.5.1.48", "numpy>=1.21.6"],
    packages=[MODULE_NAME, f'{MODULE_NAME}.models'],
    package_data={'': ['layout_cdla.onnx', '*.yaml']},
    keywords=[
        'ppstructure,layout,rapidocr,rapid_layout'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [f'{MODULE_NAME}={MODULE_NAME}.{MODULE_NAME}:main']
    }
)
