# -*- encoding: utf-8 -*-
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rapidocr.ch_ppocr_rec import TextRecognizer
from rapidocr.ch_ppocr_rec import main as rec_main
from rapidocr.utils.typings import EngineType, ModelType, OCRVersion, TaskType

DICT_URL = "https://example.com/models/ppocrv5_dict.txt"


class StubSession:
    """Session without an embedded character list, like the torch/paddle engines."""

    def have_key(self, key: str = "character") -> bool:
        return False

    @classmethod
    def get_dict_key_url(cls, file_info) -> str:
        return DICT_URL


def make_rec_cfg(model_root_dir):
    return OmegaConf.create(
        {
            "engine_type": EngineType.TORCH,
            "ocr_version": OCRVersion.PPOCRV5,
            "task_type": TaskType.REC,
            "lang_type": "ch",
            "model_type": ModelType.MOBILE,
            "rec_keys_path": None,
            "model_root_dir": model_root_dir,
        }
    )


@pytest.fixture
def downloads(monkeypatch):
    saved = []
    monkeypatch.setattr(
        rec_main.DownloadFile, "run", lambda params: saved.append(params.save_path)
    )
    return saved


def get_character_dict(cfg):
    recognizer = TextRecognizer.__new__(TextRecognizer)
    recognizer.session = StubSession()
    return recognizer.get_character_dict(cfg)


@pytest.mark.parametrize("as_type", [Path, str])
def test_rec_dict_is_downloaded_into_model_root_dir(tmp_path, downloads, as_type):
    _, dict_path = get_character_dict(make_rec_cfg(as_type(tmp_path)))

    expected = tmp_path / "ppocrv5_dict.txt"
    assert Path(dict_path) == expected
    assert downloads == [expected]


def test_rec_dict_falls_back_to_package_models_dir(downloads):
    _, dict_path = get_character_dict(make_rec_cfg(None))

    assert Path(dict_path) == rec_main.DEFAULT_MODEL_PATH / "ppocrv5_dict.txt"


def test_torch_model_path_accepts_str_model_root_dir(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from rapidocr.inference_engine.pytorch.networks import main as torch_networks

    saved = []

    def fake_download(params):
        saved.append(params.save_path)
        Path(params.save_path).touch()

    monkeypatch.setattr(torch_networks.DownloadFile, "run", fake_download)
    monkeypatch.setattr(
        torch_networks.InferSession,
        "get_model_url",
        classmethod(
            lambda cls, file_info: {
                "model_dir": "https://example.com/models/det.pth",
                "SHA256": None,
            }
        ),
    )
    cfg = OmegaConf.create(
        {
            "model_path": None,
            "model_root_dir": str(tmp_path),
            "engine_type": EngineType.TORCH,
            "ocr_version": OCRVersion.PPOCRV5,
            "task_type": TaskType.DET,
            "lang_type": "ch",
            "model_type": ModelType.MOBILE,
        }
    )

    model_path = torch_networks.ModelLoader._init_model_path(None, cfg)

    assert Path(model_path) == tmp_path / "det.pth"
    assert saved == [tmp_path / "det.pth"]
