# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import shutil
from pathlib import Path

root_dir = Path(__file__).resolve().parent
DEFAULT_CFG_PATH = root_dir / "config.yaml"


def generate_cfg(args):
    if args.save_cfg_file is None:
        args.save_cfg_file = "./default_rapidocr.yaml"

    shutil.copyfile(DEFAULT_CFG_PATH, args.save_cfg_file)
    print(f"The config file has saved in {args.save_cfg_file}")


def check_install(ocr_engine):
    img_url = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.1.0/resources/test_files/ch_en_num.jpg"
    result = ocr_engine(img_url)

    if result.txts is None or result.txts[0] != "正品促销":
        raise ValueError("The installation is incorrect!")

    print("Success! rapidocr is installed correctly!")
