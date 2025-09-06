# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr import RapidOCR

tests_dir = root_dir / "tests" / "test_files"
img_path = tests_dir / "ch_en_num.jpg"


@pytest.fixture()
def engine():
    engine = RapidOCR()
    return engine


def test_to_json(engine):
    result = engine(img_path)

    assert result.txts is not None
    result_json = result.to_json()
    assert result_json[0]["txt"] == "正品促销"


def test_to_markdown(engine):
    result = engine(img_path)

    assert result.txts is not None

    result_markdown = result.to_markdown()
    assert len(result_markdown.split("\n")) == 11
