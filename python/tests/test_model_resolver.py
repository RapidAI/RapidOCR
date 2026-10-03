# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import pytest

from rapidocr import ModelType, OCRVersion
from rapidocr.utils.model_resolver import (
    MODEL_ROUTES,
    list_supported_langs,
    route_to_model_key,
)
from rapidocr.utils.typings import TaskType
from tools.validate_model_routes import validate_model_routes


def test_model_routes_schema_is_valid():
    validate_model_routes(MODEL_ROUTES)


@pytest.mark.parametrize(
    "task_type,expected_key",
    [
        (TaskType.DET, "multi_PP-OCRv6_det_tiny"),
        (TaskType.REC, "multi_PP-OCRv6_rec_tiny"),
    ],
)
def test_ppocrv6_tiny_supports_non_japan_lang(task_type, expected_key):
    model_key = route_to_model_key(task_type, OCRVersion.PPOCRV6, "fr", ModelType.TINY)

    assert model_key == expected_key


@pytest.mark.parametrize("model_type", [ModelType.SMALL, ModelType.MEDIUM])
def test_ppocrv6_japan_supported_by_non_tiny_models(model_type):
    model_key = route_to_model_key(
        TaskType.REC, OCRVersion.PPOCRV6, "japan", model_type
    )

    assert model_key == f"multi_PP-OCRv6_rec_{model_type.value}"


@pytest.mark.parametrize("task_type", [TaskType.DET, TaskType.REC])
def test_ppocrv6_tiny_does_not_support_japan(task_type):
    with pytest.raises(ValueError, match="japan.*PP-OCRv6 tiny"):
        route_to_model_key(task_type, OCRVersion.PPOCRV6, "japan", ModelType.TINY)


def test_ppocrv6_tiny_rejects_japan_alias():
    with pytest.raises(ValueError, match="japan.*PP-OCRv6 tiny"):
        route_to_model_key(TaskType.REC, OCRVersion.PPOCRV6, "ja", ModelType.TINY)


def test_list_supported_langs_can_filter_by_model_type():
    tiny_langs = list_supported_langs(TaskType.REC, OCRVersion.PPOCRV6, ModelType.TINY)
    small_langs = list_supported_langs(
        TaskType.REC, OCRVersion.PPOCRV6, ModelType.SMALL
    )
    all_langs = list_supported_langs(TaskType.REC, OCRVersion.PPOCRV6)

    assert "japan" not in tiny_langs
    assert "japan" in small_langs
    assert "japan" in all_langs


def test_ppocrv5_rec_route_is_loaded_from_yaml():
    model_key = route_to_model_key(
        TaskType.REC, OCRVersion.PPOCRV5, "ch", ModelType.MOBILE
    )

    assert model_key == "ch_PP-OCRv5_rec_mobile"


def test_ppocrv4_cls_route_is_loaded_from_yaml():
    assert (
        route_to_model_key(TaskType.CLS, OCRVersion.PPOCRV4, "ch", ModelType.MOBILE)
        == "ch_ppocr_mobile_v2.0_cls_mobile"
    )


def test_aliases_are_scoped_to_routes():
    assert (
        route_to_model_key(TaskType.REC, OCRVersion.PPOCRV4, "ja", ModelType.MOBILE)
        == "japan_PP-OCRv4_rec_mobile"
    )


def test_list_supported_langs_includes_cls():
    assert list_supported_langs(TaskType.CLS, OCRVersion.PPOCRV5, ModelType.SERVER) == [
        "ch",
        "multi",
    ]
