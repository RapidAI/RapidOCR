# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import shlex
import shutil
import sys
from pathlib import Path

import pytest
from pytest import mark

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import LangRec, RapidOCR
from rapidocr.utils.output import RapidOCROutput
from rapidocr.main import main

tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"


@pytest.fixture()
def engine():
    engine = RapidOCR()
    return engine


@mark.parametrize("cmd,gt", [(f"--img_path {img_path}", "正品促销")])
def test_cli(capsys, cmd, gt):
    main(shlex.split(cmd))
    output = capsys.readouterr().out.strip()
    assert gt in output


@mark.parametrize("cmd", [f"download_models --config {tests_dir}/test_config.yaml"])
def test_cli_download_models(capsys, cmd):
    main(shlex.split(cmd))
    output = capsys.readouterr().out.strip()
    assert "Models downloaded to" in output

    model_dir = Path("./test_models").expanduser().resolve()
    assert model_dir.exists()
    shutil.rmtree(model_dir)


@mark.parametrize("cmd", [f"config --save_cfg_file {tests_dir}/config.yaml"])
def test_cli_config(capsys, cmd):
    main(shlex.split(cmd))
    output = capsys.readouterr().out.strip()

    assert "The config file has saved in" in output

    cfg_yaml_path = tests_dir / "config.yaml"
    assert cfg_yaml_path.exists()
    cfg_yaml_path.unlink()


@mark.parametrize("cmd", ["check"])
def test_cli_check(capsys, cmd):
    main(shlex.split(cmd))
    output = capsys.readouterr().out.strip()

    assert "Success! rapidocr is installed correctly!" in output


@mark.parametrize(
    "cmd,img_name",
    [
        (
            f"--img_path {img_path} -vis --vis_save_dir {tests_dir}",
            f"{img_path.stem}_vis.png",
        ),
        (
            f"--img_path {img_path} -vis --vis_save_dir {tests_dir} -word",
            f"{img_path.stem}_vis_single.png",
        ),
    ],
)
def test_cli_vis(cmd, img_name):
    main(shlex.split(cmd))
    vis_path = tests_dir / img_name
    assert vis_path.exists()
    vis_path.unlink()


def test_cli_lang_type():
    img_path = tests_dir / "japan.jpg"
    cmd = f"--img_path {img_path} --lang_type japan -vis --vis_save_dir {tests_dir}"
    vis_path = tests_dir / f"{img_path.stem}_vis.png"

    main(shlex.split(cmd))
    assert vis_path.exists()
    vis_path.unlink()


def test_cli_lang_type_passed_to_engine(monkeypatch):
    captured_params = {}

    class DummyRapidOCR:
        def __init__(self, params=None):
            captured_params.update(params or {})

        def __call__(self, img_path):
            return RapidOCROutput()

    monkeypatch.setattr("rapidocr.main.RapidOCR", DummyRapidOCR)

    main(shlex.split(f"--img_path {img_path} --lang_type japan"))

    assert captured_params["Rec.lang_type"] == LangRec.JAPAN


def test_cli_vis_empty_result(monkeypatch, tmp_path):
    class DummyRapidOCR:
        def __init__(self, params=None):
            pass

        def __call__(self, img_path):
            return RapidOCROutput()

    monkeypatch.setattr("rapidocr.main.RapidOCR", DummyRapidOCR)

    main(
        shlex.split(
            f"--img_path {img_path} -vis --vis_save_dir {tmp_path}"
        )
    )

    assert not (tmp_path / f"{img_path.stem}_vis.png").exists()


def test_cli_word_vis_empty_result(monkeypatch, tmp_path):
    class DummyRapidOCR:
        def __init__(self, params=None):
            pass

        def __call__(self, img_path, return_word_box=False):
            return RapidOCROutput()

    monkeypatch.setattr("rapidocr.main.RapidOCR", DummyRapidOCR)

    main(
        shlex.split(
            f"--img_path {img_path} -vis --vis_save_dir {tmp_path} -word"
        )
    )

    assert not (tmp_path / f"{img_path.stem}_vis_single.png").exists()
