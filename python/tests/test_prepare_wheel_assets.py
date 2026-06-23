# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from tools.prepare_wheel_assets import (
    ModelSpec,
    load_yaml,
    resolve_assets,
    resolve_model_specs,
    select_model_info,
)

config_path = root_dir / "rapidocr" / "config.yaml"
models_path = root_dir / "rapidocr" / "default_models.yaml"


@pytest.fixture(scope="module")
def default_config():
    return load_yaml(config_path)


@pytest.fixture(scope="module")
def model_registry():
    return load_yaml(models_path)


def test_default_wheel_assets_use_routed_ppocrv6_models(
    default_config, model_registry
):
    specs = resolve_model_specs(default_config)
    assets = resolve_assets(specs, model_registry)
    asset_paths = {asset.relative_path.as_posix() for asset in assets}

    assert asset_paths == {
        "PP-OCRv6_det_small.onnx",
        "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "PP-OCRv6_rec_small.onnx",
    }


@pytest.mark.parametrize(
    "spec,expected_model_name",
    [
        (
            ModelSpec("onnxruntime", "PP-OCRv6", "det", "ch", "small"),
            "PP-OCRv6_det_small.onnx",
        ),
        (
            ModelSpec("onnxruntime", "PP-OCRv6", "rec", "ch", "small"),
            "PP-OCRv6_rec_small.onnx",
        ),
        (
            ModelSpec("onnxruntime", "PP-OCRv4", "cls", "ch", "mobile"),
            "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        ),
    ],
)
def test_select_model_info_supports_default_model_specs(
    model_registry, spec, expected_model_name
):
    model_info = select_model_info(spec, model_registry)

    assert model_info["model_dir"].endswith(expected_model_name)
