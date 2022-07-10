# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
import setuptools
import subprocess


def get_latest_version(package_name):
    output = subprocess.run(["pip", "index", "versions", package_name],
                            capture_output=True)
    output = output.stdout.decode('utf-8')
    if output:
        output = list(filter(lambda x: len(x) > 0, output.split('\n')))
        latest_version = output[-1].split(':')[1].strip()
        return latest_version
    else:
        return None


def version_add_one(version, add_loc=-1):
    if version:
        version_list = version.split('.')
        mini_version = str(int(version_list[add_loc]) + 1)
        version_list[add_loc] = mini_version
        new_version = '.'.join(version_list)
        return new_version
    else:
        return None


def get_readme():
    root_dir = Path(__file__).resolve().parent.parent
    readme_path = str(root_dir / 'docs' / 'doc_whl_en.md')
    print(readme_path)
    with open(readme_path, 'r') as f:
        readme = f.read()
    return readme

module_name = 'rapidocr_onnxruntime'
latest_version = get_latest_version(module_name)
version_num = version_add_one(latest_version)


setuptools.setup(
    name=module_name,
    version=version_num,
    platforms="Windows & Linux & Mac",
    description="RapidOCR",
    long_description=get_readme(),
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
    packages=setuptools.find_packages(where=module_name),
    keywords=[
        'ocr textdetection textrecognition crnn east star-net rosetta ocrlite db chineseocr chinesetextdetection chinesetextrecognition'
    ],
)

