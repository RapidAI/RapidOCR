# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import hashlib
import importlib
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from tqdm import tqdm


def get_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    with open(file_path, "rb") as file:
        sha_signature = hashlib.sha256()
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            sha_signature.update(chunk)

    return sha_signature.hexdigest()


def save_img(save_path: Union[str, Path], img: np.ndarray):
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_path), img)


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


def download_file(url: str, save_path: Union[str, Path], logger):
    if not Path(save_path).parent.exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(save_path).exists():
        logger.info("File already exists in %s", save_path)
        return

    logger.info("Downloading %s to %s", url, save_path)
    try:
        response = requests.get(url, stream=True, timeout=60)
    except Exception as e:
        raise e

    status_code = response.status_code
    if status_code != 200:
        raise DownloadFileException(f"Something went wrong while downloading {url}")

    total_size_in_bytes = int(response.headers.get("content-length", 1))
    block_size = 1024  # 1 Kibibyte
    with tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True) as pb:
        with open(save_path, "wb") as file:
            for data in response.iter_content(block_size):
                pb.update(len(data))
                file.write(data)


class DownloadFileException(Exception):
    pass
