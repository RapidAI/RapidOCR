# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path
from typing import Union

from ..inference_engine.base import FileInfo, InferSession
from .download_file import DownloadFile, DownloadFileInput
from .log import logger
from .parse_parameters import ParseParams
from .typings import EngineType, TaskType
from .utils import mkdir

root_dir = Path(__file__).resolve().parent.parent


def download_models(config_path: Union[str, Path]) -> None:
    cfg = ParseParams.load(config_path)

    model_root_dir = cfg.Global.get("model_root_dir")
    if model_root_dir is None:
        model_root_dir = root_dir / "models"

    model_root_dir = Path(model_root_dir).expanduser().resolve()
    mkdir(model_root_dir)

    use_det = cfg.Global.get("use_det", True)
    use_cls = cfg.Global.get("use_cls", True)
    use_rec = cfg.Global.get("use_rec", True)
    if not (use_det or use_cls or use_rec):
        raise ValueError(
            "Config has use_det, use_cls, use_rec all false; at least one must be True."
        )

    if use_det:
        download_task(
            model_root_dir,
            FileInfo(
                cfg.Det.engine_type,
                cfg.Det.ocr_version,
                cfg.Det.task_type,
                cfg.Det.lang_type,
                cfg.Det.model_type,
            ),
        )

    if use_cls:
        download_task(
            model_root_dir,
            FileInfo(
                cfg.Cls.engine_type,
                cfg.Cls.ocr_version,
                cfg.Cls.task_type,
                cfg.Cls.lang_type,
                cfg.Cls.model_type,
            ),
        )

    if use_rec:
        download_task(
            model_root_dir,
            FileInfo(
                cfg.Rec.engine_type,
                cfg.Rec.ocr_version,
                cfg.Rec.task_type,
                cfg.Rec.lang_type,
                cfg.Rec.model_type,
            ),
        )

    print(f"Models downloaded to {model_root_dir}")
    print(
        f"Please initialize RapidOCR using the {config_path} configuration file to ensure the model used is consistent with the pre-downloaded one."
    )


def download_task(cache_dir: Path, file_info: FileInfo) -> None:
    info = InferSession.get_model_url(file_info)
    engine_type = file_info.engine_type

    if engine_type != EngineType.PADDLE:
        download_single_file(cache_dir, info)
    else:
        download_paddle_bundle(cache_dir, info)

    if file_info.task_type == TaskType.REC:
        dict_url = info.get("dict_url")
        if dict_url:
            DownloadFile.run(
                DownloadFileInput(
                    file_url=dict_url,
                    save_path=cache_dir / Path(dict_url).name,
                    logger=logger,
                    sha256=None,
                )
            )


def download_single_file(cache_dir: Path, info: dict) -> None:
    model_url = info["model_dir"]
    save_path = cache_dir / Path(model_url).name
    DownloadFile.run(
        DownloadFileInput(
            file_url=model_url,
            save_path=save_path,
            logger=logger,
            sha256=info.get("SHA256"),
        )
    )


def download_paddle_bundle(cache_dir: Path, info: dict) -> None:
    model_url_dir = info["model_dir"].rstrip("/")
    save_model_dir = cache_dir / Path(model_url_dir).name
    for name, sha in info.items():
        if name in ("model_dir", "dict_url"):
            continue

        DownloadFile.run(
            DownloadFileInput(
                file_url=f"{model_url_dir}/{name}",
                save_path=save_model_dir / name,
                logger=logger,
                sha256=sha,
            )
        )
