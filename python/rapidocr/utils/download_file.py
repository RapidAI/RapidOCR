# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import requests
from tqdm import tqdm

from .utils import get_file_sha256


@dataclass
class DownloadFileInput:
    file_url: str
    save_path: Union[str, Path]
    sha256: Optional[str] = None


class DownloadFile:
    @classmethod
    def run(cls, input_params: DownloadFileInput, logger):
        save_path = Path(input_params.save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            if input_params.sha256 is not None and cls.check_file_sha256(
                save_path, input_params.sha256
            ):
                logger.info("File already exists in %s", save_path)
                return
            logger.warning("%s is damaged and needs to be downloaded again.", save_path)

        logger.info("Downloading %s to %s", input_params.file_url, save_path)
        try:
            response = requests.get(input_params.file_url, stream=True, timeout=60)
        except Exception as e:
            raise e

        status_code = response.status_code
        if status_code != 200:
            raise DownloadFileException(
                f"Something went wrong while downloading {input_params.file_url}"
            )

        total_size_in_bytes = int(response.headers.get("content-length", 1))
        block_size = 1024  # 1 Kibibyte
        with tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True) as pb:
            with open(save_path, "wb") as file:
                for data in response.iter_content(block_size):
                    pb.update(len(data))
                    file.write(data)

    @staticmethod
    def check_file_sha256(file_path: Union[str, Path], gt_sha256: str) -> bool:
        return get_file_sha256(file_path) == gt_sha256


class DownloadFileException(Exception):
    pass
