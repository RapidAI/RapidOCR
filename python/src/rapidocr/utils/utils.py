"""Small helpers shared with the 3.x download path."""

import hashlib
from pathlib import Path
from typing import Union


def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    with open(file_path, "rb") as file:
        sha_signature = hashlib.sha256()
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            sha_signature.update(chunk)
    return sha_signature.hexdigest()
