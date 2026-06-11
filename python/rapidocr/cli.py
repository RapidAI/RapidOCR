# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import shutil
from pathlib import Path
from typing import List

root_dir = Path(__file__).resolve().parent
DEFAULT_CFG_PATH = root_dir / "config.yaml"
REQUIRED_PACKAGE_FILES = [
    DEFAULT_CFG_PATH,
    root_dir / "default_models.yaml",
    root_dir / "inference_engine" / "pytorch" / "networks" / "arch_config.yaml",
]


def generate_cfg(args):
    if args.save_cfg_file is None:
        args.save_cfg_file = "./default_rapidocr.yaml"

    shutil.copyfile(DEFAULT_CFG_PATH, args.save_cfg_file)
    print(f"The config file has saved in {args.save_cfg_file}")


def check_required_files() -> None:
    missing_files: List[str] = [
        str(file_path) for file_path in REQUIRED_PACKAGE_FILES if not file_path.is_file()
    ]
    if missing_files:
        missing_info = "\n".join(f"  - {file_path}" for file_path in missing_files)
        raise FileNotFoundError(
            "RapidOCR installation is missing required package files:\n"
            f"{missing_info}"
        )


def check_install(ocr_engine, check_files: bool = True):
    if check_files:
        check_required_files()

    img_url = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.1.0/resources/test_files/ch_en_num.jpg"
    result = ocr_engine(img_url)

    if result.txts is None or result.txts[0] != "正品促销":
        raise ValueError("The installation is incorrect!")

    print("Success! rapidocr is installed correctly!")
