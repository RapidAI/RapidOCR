# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytest

from rapidocr.utils.typings import OCRVersion

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import EngineType, LangRec, LoadImageError, ModelType, RapidOCR
from rapidocr.main import main
from rapidocr.utils.logger import Logger

tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"
logger = Logger(logger_name=__name__).get_log()


@pytest.fixture()
def engine():
    engine = RapidOCR()
    return engine


def get_engine(params: Optional[Dict[str, Any]] = None):
    if params:
        engine = RapidOCR(params=params)
        return engine

    engine = RapidOCR()
    return engine


@pytest.mark.parametrize(
    "engine_type",
    [EngineType.ONNXRUNTIME, EngineType.PADDLE, EngineType.OPENVINO],
)
def test_ppocrv5_rec_mobile(engine_type):
    engine = RapidOCR(
        params={
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.engine_type": engine_type,
        }
    )
    img_path = tests_dir / "text_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)

    assert result.txts is not None
    assert result.txts[0] == "韩国小馆"


def test_ppocrv5_det_mobile():
    engine = RapidOCR(
        params={
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
        }
    )
    img_path = tests_dir / "ch_en_num.jpg"
    result = engine(img_path, use_det=True, use_cls=False, use_rec=False)

    assert result.boxes is not None
    assert len(result.boxes) == 17


def test_ch_doc_server():
    engine = RapidOCR(
        params={"Rec.lang_type": LangRec.CH_DOC, "Rec.model_type": ModelType.SERVER}
    )

    img_path = tests_dir / "ch_doc_server.png"
    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == "嫖娼"


def test_vis_only_det(engine):
    img_path = tests_dir / "ch_en_num.jpg"
    result = engine(img_path, use_det=True, use_cls=False, use_rec=False)
    vis_img = result.vis()
    assert vis_img.shape[2] == 3


def test_vis_only_rec(engine):
    img_path = tests_dir / "text_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)
    vis_img = result.vis()
    assert vis_img.shape[2] == 3


def test_vis_only_cls(engine):
    img_path = tests_dir / "text_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=False)
    vis_img = result.vis()
    assert vis_img.shape[2] == 3


def test_vis_det_cls_rec(engine):
    img_path = tests_dir / "ch_en_num.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=False)
    vis_img = result.vis()
    assert vis_img.shape[2] == 3


def test_full_black_img(engine):
    img_path = tests_dir / "empty_black.jpg"
    result = engine(img_path)
    assert result.img is None
    assert result.boxes is None


def test_img_url_input(engine):
    img_url = "https://github.com/RapidAI/RapidOCR/blob/a9bb7c1f44b6e00556ada90ac588f020d7637c4b/python/tests/test_files/ch_en_num.jpg?raw=true"
    result = engine(img_url)
    assert result.txts is not None
    assert result.txts[0] == "正品促销"


def test_server_rec():
    engine = RapidOCR(
        params={"Rec.lang_type": LangRec.CH, "Rec.model_type": ModelType.SERVER}
    )
    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == "正品促销"


@pytest.mark.parametrize("cmd,gt", [(f"--img_path {img_path}", "正品促销")])
def test_cli(capsys, cmd, gt):
    main(shlex.split(cmd))
    output = capsys.readouterr().out.strip()
    assert gt in output


@pytest.mark.parametrize("cmd", [f"config --save_cfg_file {tests_dir}/config.yaml"])
def test_cli_config(capsys, cmd):
    main(shlex.split(cmd))
    output = capsys.readouterr().out.strip()

    assert "The config file has saved in" in output

    cfg_yaml_path = tests_dir / "config.yaml"
    assert cfg_yaml_path.exists()
    cfg_yaml_path.unlink()


@pytest.mark.parametrize("cmd", ["check"])
def test_cli_check(capsys, cmd):
    main(shlex.split(cmd))
    output = capsys.readouterr().out.strip()

    assert "Success! rapidocr is installed correctly!" in output


@pytest.mark.parametrize(
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


def test_korean_lang():
    engine = get_engine(
        params={"Rec.lang_type": LangRec.KOREAN, "Rec.model_type": ModelType.MOBILE}
    )
    img_path = tests_dir / "korean.jpg"
    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == "베이징차오양"


def test_en_lang():
    engine = get_engine(
        params={"Rec.lang_type": LangRec.EN, "Rec.model_type": ModelType.MOBILE}
    )
    img_path = tests_dir / "en.jpg"
    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == "3"


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="does not run on macOS")
def test_engine_openvino():
    engine = get_engine(
        params={
            "Det.engine_type": EngineType.OPENVINO,
            "Cls.engine_type": EngineType.OPENVINO,
            "Rec.engine_type": EngineType.OPENVINO,
        }
    )

    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == "正品促销"


def test_engine_paddle():
    engine = get_engine(
        params={
            "Det.engine_type": EngineType.PADDLE,
            "Cls.engine_type": EngineType.PADDLE,
            "Rec.engine_type": EngineType.PADDLE,
        }
    )

    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == "正品促销"


def test_engine_torch():
    engine = get_engine(
        params={
            "Det.engine_type": EngineType.TORCH,
            "Cls.engine_type": EngineType.TORCH,
            "Rec.engine_type": EngineType.TORCH,
        }
    )

    result = engine(img_path)
    assert result.txts is not None
    assert result.txts[0] == "正品促销"


def test_long_img(engine):
    img_url = "https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/long.jpeg"
    result = engine(img_url)

    assert result is not None
    assert len(result.boxes) >= 53


def test_mode_one_img(engine):
    img_path = tests_dir / "issue_170.png"
    result = engine(img_path)
    assert result.txts[0] == "TEST"


@pytest.mark.parametrize(
    "img_name,gt",
    [
        (
            "black_font_color_transparent.png",
            "我是中国人",
        ),
        (
            "white_font_color_transparent.png",
            "我是中国人",
        ),
    ],
)
def test_transparent_img(engine, img_name: str, gt: str):
    img_path = tests_dir / img_name
    result = engine(img_path)
    assert result.txts[0] == gt


@pytest.mark.parametrize(
    "img_name,gt_len,gt_first_len",
    [
        (
            "test_letterbox_like.jpg",
            2,
            "A：：取决于所使用的执行提供者，它可能没有完全支持模型中的所有操作。回落到CPU操作可能会导致性能速度的下降。此外，即使一个操作是由CUDAeXecution",
        ),
        ("test_without_det.jpg", 1, "在中国作家协会第三届儿童文学"),
    ],
)
def test_letterbox_like(engine, img_name, gt_len, gt_first_len):
    img_path = tests_dir / img_name
    result = engine(img_path)

    assert len(result) == gt_len
    assert result.txts[0].lower() == gt_first_len.lower()


def test_only_det(engine):
    result = engine(img_path, use_det=True, use_cls=False, use_rec=False)
    assert len(result) == 18


def test_only_cls(engine):
    img_path = tests_dir / "text_cls.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=False)
    assert len(result) == 1
    assert result.cls_res[0][0] == "0"


def test_only_rec(engine):
    img_path = tests_dir / "text_rec.jpg"
    result = engine(img_path, use_det=False, use_cls=False, use_rec=True)
    assert len(result) == 1
    assert result.txts[0] == "韩国小馆"


def test_det_rec(engine):
    result = engine(img_path, use_det=True, use_cls=False, use_rec=True)
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_cls_rec(engine):
    img_path = tests_dir / "text_cls.jpg"
    result = engine(img_path, use_det=False, use_cls=True, use_rec=True)

    assert result is not None
    assert len(result) == 1
    assert result.txts[0] == "韩国小馆"


def test_det_cls_rec(engine):
    img = cv2.imread(str(img_path))

    result = engine(img)
    assert result is not None
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_empty(engine):
    img = None
    with pytest.raises(LoadImageError) as exc_info:
        engine(img)
        raise LoadImageError
    assert exc_info.type is LoadImageError


def test_zeros(engine):
    img = np.zeros([640, 640, 3], np.uint8)
    result = engine(img)
    assert result.boxes is None


def test_input_str(engine):
    result = engine(str(img_path))
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_input_bytes(engine):
    with open(img_path, "rb") as f:
        result = engine(f.read())
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_input_path(engine):
    result = engine(img_path)
    assert len(result) == 18
    assert result.txts[0] == "正品促销"


def test_input_parameters(engine):
    result = engine(img_path, text_score=1.0)
    assert result.boxes is None


def test_input_three_ndim_two_channel(engine):
    img_npy = tests_dir / "two_dim_image.npy"
    image_array = np.load(str(img_npy))
    result = engine(image_array)

    assert result is not None
    assert len(result) == 1
    assert result.txts[0] == "TREND PLOT REPORT"


def test_input_three_ndim_one_channel(engine):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:, :, 0]
    img = img[..., None]  # (H, W, 1)

    result = engine(img)
    assert len(result) >= 17


@pytest.mark.parametrize(
    "img_name,words",
    [
        (
            "black_font_color_transparent.png",
            ("我", "是", "中", "国", "人"),
        ),
        (
            "text_vertical_words.png",
            ("已", "取", "之", "時", "不", "參", "一", "人", "見", "而"),
        ),
    ],
)
def test_cn_word_ocr(engine, img_name: str, words: List[str]):
    img_path = tests_dir / img_name
    result = engine(img_path, return_word_box=True)
    txts, _, _ = list(zip(*result.word_results[0]))
    assert txts == words


@pytest.mark.parametrize(
    "img_name,words",
    [("issue_170.png", "TEST"), ("return_word_debug.jpg", "3F1")],
)
def test_en_word_ocr(engine, img_name: str, words: str):
    img_path = tests_dir / img_name
    result = engine(img_path, return_word_box=True)
    txts, _, _ = list(zip(*result.word_results[0]))
    assert txts[0] == words


def test_en_return_single_char_box(engine):
    img_path = tests_dir / "en.jpg"
    result = engine(img_path, return_word_box=True, return_single_char_box=True)
    txts, _, _ = list(zip(*result.word_results[0]))
    assert txts[:3] == ("3",)
