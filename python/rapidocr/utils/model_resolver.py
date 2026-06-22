# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, List, Mapping, Optional, Union

from .typings import ModelType, OCRVersion, TaskType


@dataclass(frozen=True)
class ModelRoute:
    model_key_template: str
    supported_langs_by_model_type: Mapping[ModelType, FrozenSet[str]]
    aliases: Mapping[str, str] = field(default_factory=dict)

    def supported_langs(self, model_type: ModelType) -> FrozenSet[str]:
        return self.supported_langs_by_model_type.get(model_type, frozenset())


COMMON_LANG_ALIASES = {
    "zh": "ch",
    "zh_cn": "ch",
    "zh-cn": "ch",
    "zh_tw": "chinese_cht",
    "zh-tw": "chinese_cht",
    "ja": "japan",
    "jp": "japan",
    "ko": "korean",
}

PP_OCRV6_LANGS = frozenset(
    {
        "ch",
        "chinese_cht",
        "en",
        "japan",
        "af",
        "az",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "es",
        "et",
        "eu",
        "fi",
        "fr",
        "ga",
        "gl",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lb",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pi",
        "pt",
        "qu",
        "rm",
        "ro",
        "rs_latin",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
        "french",
        "german",
    }
)
PP_OCRV6_TINY_LANGS = PP_OCRV6_LANGS - {"japan"}
PP_OCRV6_LANGS_BY_MODEL_TYPE = {
    ModelType.TINY: PP_OCRV6_TINY_LANGS,
    ModelType.SMALL: PP_OCRV6_LANGS,
    ModelType.MEDIUM: PP_OCRV6_LANGS,
}

MODEL_ROUTES = {
    TaskType.DET: {
        OCRVersion.PPOCRV6: ModelRoute(
            model_key_template="multi_PP-OCRv6_det_{model_type}",
            supported_langs_by_model_type=PP_OCRV6_LANGS_BY_MODEL_TYPE,
            aliases=COMMON_LANG_ALIASES,
        ),
    },
    TaskType.REC: {
        OCRVersion.PPOCRV6: ModelRoute(
            model_key_template="multi_PP-OCRv6_rec_{model_type}",
            supported_langs_by_model_type=PP_OCRV6_LANGS_BY_MODEL_TYPE,
            aliases=COMMON_LANG_ALIASES,
        ),
    },
}


def normalize_lang(lang_type: Union[Enum, str]) -> str:
    if isinstance(lang_type, Enum):
        lang = lang_type.value
    else:
        lang = str(lang_type)

    return lang.strip().lower()


def resolve_model_key(
    task_type: TaskType,
    ocr_version: OCRVersion,
    lang_type: Union[Enum, str],
    model_type: ModelType,
) -> Optional[str]:
    route = MODEL_ROUTES.get(task_type, {}).get(ocr_version)
    if route is None:
        return None

    lang = normalize_lang(lang_type)
    lang = route.aliases.get(lang, lang)
    supported_langs = route.supported_langs(model_type)

    if lang not in supported_langs:
        raise ValueError(
            f"Unsupported {task_type.value}.lang_type={lang!r} for "
            f"{ocr_version.value} {model_type.value} model."
        )

    return route.model_key_template.format(model_type=model_type.value)


def list_supported_langs(
    task_type: TaskType,
    ocr_version: OCRVersion,
    model_type: Optional[ModelType] = None,
) -> List[str]:
    route = MODEL_ROUTES.get(task_type, {}).get(ocr_version)
    if route is None:
        return []

    if model_type is not None:
        return sorted(route.supported_langs(model_type))

    supported_langs = set()
    for langs in route.supported_langs_by_model_type.values():
        supported_langs.update(langs)
    return sorted(supported_langs)
