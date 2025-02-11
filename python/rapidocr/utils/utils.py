# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm


def download_file(url: str, save_path: Union[str, Path]):
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
