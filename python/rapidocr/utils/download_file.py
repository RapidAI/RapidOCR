# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import logging
import sys
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
    logger: logging.Logger
    sha256: Optional[str] = None


class DownloadFile:
    BLOCK_SIZE = 1024  # 1 KiB
    REQUEST_TIMEOUT = 60

    @classmethod
    def run(cls, input_params: DownloadFileInput):
        save_path = Path(input_params.save_path)

        logger = input_params.logger
        cls._ensure_parent_dir_exists(save_path)
        if cls._should_skip_download(save_path, input_params.sha256, logger):
            return

        response = cls._make_http_request(input_params.file_url, logger)
        cls._save_response_with_progress(response, save_path, logger)

    @staticmethod
    def _ensure_parent_dir_exists(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _should_skip_download(
        cls, path: Path, expected_sha256: Optional[str], logger: logging.Logger
    ) -> bool:
        if not path.exists():
            return False

        if expected_sha256 is None:
            logger.info("File exists (no checksum verification): %s", path)
            return True

        if cls.check_file_sha256(path, expected_sha256):
            logger.info("File exists and is valid: %s", path)
            return True

        logger.warning("File exists but is invalid, redownloading: %s", path)
        return False

    @classmethod
    def _make_http_request(cls, url: str, logger: logging.Logger) -> requests.Response:
        logger.info("Initiating download: %s", url)
        try:
            response = requests.get(url, stream=True, timeout=cls.REQUEST_TIMEOUT)
            response.raise_for_status()  # Raises HTTPError for 4XX/5XX
            return response
        except requests.RequestException as e:
            logger.error("Download failed: %s", url)
            raise DownloadFileException(f"Failed to download {url}") from e

    @classmethod
    def _save_response_with_progress(
        cls, response: requests.Response, save_path: Path, logger: logging.Logger
    ) -> None:
        total_size = int(response.headers.get("content-length", 0))
        logger.info("Download size: %.2fMB", total_size / 1024 / 1024)

        with tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            disable=not cls.check_is_atty(),
        ) as progress_bar:
            with open(save_path, "wb") as output_file:
                for chunk in response.iter_content(chunk_size=cls.BLOCK_SIZE):
                    progress_bar.update(len(chunk))
                    output_file.write(chunk)

        logger.info("Successfully saved to: %s", save_path)

    @staticmethod
    def check_file_sha256(file_path: Union[str, Path], gt_sha256: str) -> bool:
        return get_file_sha256(file_path) == gt_sha256

    @staticmethod
    def check_is_atty() -> bool:
        try:
            is_interactive = sys.stderr.isatty()
        except AttributeError:
            return False
        return is_interactive


class DownloadFileException(Exception):
    pass
