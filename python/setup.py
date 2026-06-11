# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import shutil
import sys
from pathlib import Path
from typing import List, Union

import setuptools
from get_pypi_latest_version import GetPyPiLatestVersion

from rapidocr.utils.download_models import download_models


def read_txt(txt_path: Union[Path, str]) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        data = [v.rstrip("\n") for v in f]
    return data


def get_readme():
    root_dir = Path(__file__).resolve().parent.parent
    readme_path = str(root_dir / "docs" / "doc_whl_rapidocr.md")
    print(readme_path)
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    return readme


def collect_package_files(package_root: Path, pattern: str) -> List[str]:
    return sorted(
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob(pattern)
        if path.is_file()
    )


download_models()

MODULE_NAME = "rapidocr"
PACKAGE_DIR = Path(MODULE_NAME)
PACKAGE_ROOT = PACKAGE_DIR
WRAP_DIR = Path(f"wrap_temp/{MODULE_NAME}")
NEED_WRAP = "bdist_wheel" in sys.argv

if NEED_WRAP:
    if WRAP_DIR.exists():
        shutil.rmtree(WRAP_DIR)
    WRAP_DIR.mkdir(parents=True)

    dest_dir = WRAP_DIR / MODULE_NAME
    shutil.copytree(PACKAGE_DIR, dest_dir)

    with open(WRAP_DIR / "__init__.py", "w", encoding="utf-8") as f:
        f.write("from .rapidocr.main import RapidOCR, VisRes\n")

    PACKAGE_DIR = WRAP_DIR
    PACKAGE_ROOT = PACKAGE_DIR / MODULE_NAME

PACKAGE_DATA = collect_package_files(PACKAGE_ROOT, "*.yaml")
PACKAGE_DATA.extend(
    [
        "models/ch_PP-OCRv4_det_mobile.onnx",
        "models/ch_PP-OCRv4_rec_mobile.onnx",
        "models/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    ]
)

obtainer = GetPyPiLatestVersion()
try:
    latest_version = obtainer(MODULE_NAME)
except Exception as e:
    latest_version = "0.0.0"
VERSION_NUM = obtainer.version_add_one(latest_version, add_patch=True)

if len(sys.argv) > 2:
    match_str = " ".join(sys.argv[2:])
    matched_versions = obtainer.extract_version(match_str)
    if matched_versions:
        VERSION_NUM = matched_versions
sys.argv = sys.argv[:2]

project_urls = {
    "Documentation": "https://rapidai.github.io/RapidOCRDocs",
    "Changelog": "https://github.com/RapidAI/RapidOCR/releases",
}

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    platforms="Any",
    description="Awesome OCR Library",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    author="SWHL",
    author_email="liekkaskono@163.com",
    url="https://github.com/RapidAI/RapidOCR",
    project_urls=project_urls,
    license="Apache-2.0",
    include_package_data=True,
    install_requires=read_txt("requirements.txt"),
    package_dir={"": str(PACKAGE_DIR)},
    packages=setuptools.find_namespace_packages(where=str(PACKAGE_DIR)),
    package_data={MODULE_NAME: PACKAGE_DATA},
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
    python_requires=">=3.6,<4",
    entry_points={
        "console_scripts": [f"{MODULE_NAME}={MODULE_NAME}.main:main"],
    },
)

if NEED_WRAP and Path("wrap_temp").exists():
    shutil.rmtree("wrap_temp")
