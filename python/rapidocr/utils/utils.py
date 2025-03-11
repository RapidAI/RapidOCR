# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import importlib
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import requests
from tqdm import tqdm


def is_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception as e:
        return False


def import_package(name, package=None):
    try:
        module = importlib.import_module(name, package=package)
        return module
    except ModuleNotFoundError:
        return None


def download_file(url: str, save_path: Union[str, Path], logger=None):
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(save_path).exists():
        if logger is not None:
            logger.info("File already exists in %s", save_path)
        else:
            print(f"File already exists in {save_path}")
        return

    if logger is not None:
        logger.info("Downloading file from %s to %s", url, save_path)
    else:
        print(f"Downloading file from {url} to {save_path}")

    response = requests.get(url, stream=True, timeout=60)
    status_code = response.status_code

    if status_code != 200:
        raise DownloadFileError("Something went wrong while downloading models")

    total_size_in_bytes = int(response.headers.get("content-length", 1))
    block_size = 1024  # 1 Kibibyte
    with tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True) as pb:
        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                pb.update(len(data))
                file.write(data)


class DownloadFileError(Exception):
    pass
