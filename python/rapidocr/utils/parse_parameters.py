# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
InputType = Union[str, np.ndarray, bytes, Path, Image.Image]

DEFAULT_OCR_MODEL_VERSION = "PP-OCRv4"
SUPPORT_OCR_MODEL_VERSION = ["PP-OCR", "PP-OCRv2", "PP-OCRv3", "PP-OCRv4"]
MODEL_URLS = {
    "OCR": {
        "PP-OCRv4": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
                },
                "ml": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar"
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/en_dict.txt",
                },
                "korean": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ta_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/te_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/ka_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/arabic_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/devanagari_PP-OCRv4_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/devanagari_dict.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
        "PP-OCRv3": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
                },
                "ml": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar"
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/en_dict.txt",
                },
                "korean": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/te_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/devanagari_dict.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
        "PP-OCRv2": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar",
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                }
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
        "PP-OCR": {
            "det": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar",
                },
                "structure": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar"
                },
            },
            "rec": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
                },
                "en": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/en_dict.txt",
                },
                "french": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/french_dict.txt",
                },
                "german": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/german_dict.txt",
                },
                "korean": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/korean_dict.txt",
                },
                "japan": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/japan_dict.txt",
                },
                "chinese_cht": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/chinese_cht_dict.txt",
                },
                "ta": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ta_dict.txt",
                },
                "te": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/te_dict.txt",
                },
                "ka": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/ka_dict.txt",
                },
                "latin": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/latin_dict.txt",
                },
                "arabic": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/arabic_dict.txt",
                },
                "cyrillic": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/cyrillic_dict.txt",
                },
                "devanagari": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar",
                    "dict_path": "./ppocr/utils/dict/devanagari_dict.txt",
                },
                "structure": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar",
                    "dict_path": "ppocr/utils/dict/table_dict.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                }
            },
        },
    }
}


def parse_lang(lang):
    latin_lang = [
        "af",
        "az",
        "bs",
        "cs",
        "cy",
        "da",
        "de",
        "es",
        "et",
        "fr",
        "ga",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
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
    ]
    arabic_lang = ["ar", "fa", "ug", "ur"]
    cyrillic_lang = [
        "ru",
        "rs_cyrillic",
        "be",
        "bg",
        "uk",
        "mn",
        "abq",
        "ady",
        "kbd",
        "ava",
        "dar",
        "inh",
        "che",
        "lbe",
        "lez",
        "tab",
    ]
    devanagari_lang = [
        "hi",
        "mr",
        "ne",
        "bh",
        "mai",
        "ang",
        "bho",
        "mah",
        "sck",
        "new",
        "gom",
        "sa",
        "bgc",
    ]
    if lang in latin_lang:
        lang = "latin"
    elif lang in arabic_lang:
        lang = "arabic"
    elif lang in cyrillic_lang:
        lang = "cyrillic"
    elif lang in devanagari_lang:
        lang = "devanagari"

    assert (
        lang in MODEL_URLS["OCR"][DEFAULT_OCR_MODEL_VERSION]["rec"]
    ), "param lang must in {}, but got {}".format(
        MODEL_URLS["OCR"][DEFAULT_OCR_MODEL_VERSION]["rec"].keys(), lang
    )

    if lang == "ch":
        det_lang = "ch"
    elif lang in ["en", "latin"]:
        det_lang = "en"
    else:
        det_lang = "ml"

    return det_lang, lang


def update_model_path(config: DictConfig) -> DictConfig:
    key = "model_path"
    config["Det"][key] = str(root_dir / config["Det"][key])
    config["Rec"][key] = str(root_dir / config["Rec"][key])
    config["Cls"][key] = str(root_dir / config["Cls"][key])
    return config


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", "--img_path", type=str, default=None, required=True)
    parser.add_argument("-p", "--print_cost", action="store_true", default=False)

    global_group = parser.add_argument_group(title="Global")
    global_group.add_argument("--text_score", type=float, default=0.5)

    global_group.add_argument("--no_det", action="store_true", default=False)
    global_group.add_argument("--no_cls", action="store_true", default=False)
    global_group.add_argument("--no_rec", action="store_true", default=False)

    global_group.add_argument("--print_verbose", action="store_true", default=False)
    global_group.add_argument("--min_height", type=int, default=30)
    global_group.add_argument("--width_height_ratio", type=int, default=8)
    global_group.add_argument("--max_side_len", type=int, default=2000)
    global_group.add_argument("--min_side_len", type=int, default=30)
    global_group.add_argument("--return_word_box", action="store_true", default=False)

    global_group.add_argument("--intra_op_num_threads", type=int, default=-1)
    global_group.add_argument("--inter_op_num_threads", type=int, default=-1)

    det_group = parser.add_argument_group(title="Det")
    det_group.add_argument("--det_use_cuda", action="store_true", default=False)
    det_group.add_argument("--det_use_dml", action="store_true", default=False)
    det_group.add_argument("--det_model_path", type=str, default=None)
    det_group.add_argument("--det_limit_side_len", type=float, default=736)
    det_group.add_argument(
        "--det_limit_type", type=str, default="min", choices=["max", "min"]
    )
    det_group.add_argument("--det_thresh", type=float, default=0.3)
    det_group.add_argument("--det_box_thresh", type=float, default=0.5)
    det_group.add_argument("--det_unclip_ratio", type=float, default=1.6)
    det_group.add_argument(
        "--det_donot_use_dilation", action="store_true", default=False
    )
    det_group.add_argument(
        "--det_score_mode", type=str, default="fast", choices=["slow", "fast"]
    )

    cls_group = parser.add_argument_group(title="Cls")
    cls_group.add_argument("--cls_use_cuda", action="store_true", default=False)
    cls_group.add_argument("--cls_use_dml", action="store_true", default=False)
    cls_group.add_argument("--cls_model_path", type=str, default=None)
    cls_group.add_argument("--cls_image_shape", type=list, default=[3, 48, 192])
    cls_group.add_argument("--cls_label_list", type=list, default=["0", "180"])
    cls_group.add_argument("--cls_batch_num", type=int, default=6)
    cls_group.add_argument("--cls_thresh", type=float, default=0.9)

    rec_group = parser.add_argument_group(title="Rec")
    rec_group.add_argument("--rec_use_cuda", action="store_true", default=False)
    rec_group.add_argument("--rec_use_dml", action="store_true", default=False)
    rec_group.add_argument("--rec_model_path", type=str, default=None)
    rec_group.add_argument("--rec_keys_path", type=str, default=None)
    rec_group.add_argument("--rec_img_shape", type=list, default=[3, 48, 320])
    rec_group.add_argument("--rec_batch_num", type=int, default=6)

    vis_group = parser.add_argument_group(title="Visual Result")
    vis_group.add_argument("-vis", "--vis_res", action="store_true", default=False)
    vis_group.add_argument(
        "--vis_font_path",
        type=str,
        default=None,
        help="When -vis is True, the font_path must have value.",
    )
    vis_group.add_argument(
        "--vis_save_path",
        type=str,
        default=".",
        help="The directory of saving the vis image.",
    )

    args = parser.parse_args()
    return args


class UpdateParameters:
    def __init__(self) -> None:
        pass

    def parse_kwargs(self, **kwargs):
        global_dict, det_dict, cls_dict, rec_dict = {}, {}, {}, {}
        for k, v in kwargs.items():
            if k.startswith("det"):
                k = k.split("det_")[1]
                if k == "donot_use_dilation":
                    k = "use_dilation"
                    v = not v

                det_dict[k] = v
            elif k.startswith("cls"):
                cls_dict[k] = v
            elif k.startswith("rec"):
                rec_dict[k] = v
            else:
                global_dict[k] = v
        return global_dict, det_dict, cls_dict, rec_dict

    def __call__(self, config, **kwargs):
        global_dict, det_dict, cls_dict, rec_dict = self.parse_kwargs(**kwargs)
        new_config = {
            "Global": self.update_global_params(config["Global"], global_dict),
            "Det": self.update_params(
                config["Det"],
                det_dict,
                "det_",
                ["det_model_path", "det_use_cuda", "det_use_dml"],
            ),
            "Cls": self.update_params(
                config["Cls"],
                cls_dict,
                "cls_",
                ["cls_label_list", "cls_model_path", "cls_use_cuda", "cls_use_dml"],
            ),
            "Rec": self.update_params(
                config["Rec"],
                rec_dict,
                "rec_",
                ["rec_model_path", "rec_use_cuda", "rec_use_dml"],
            ),
        }

        update_params = ["intra_op_num_threads", "inter_op_num_threads"]
        new_config = self.update_global_to_module(
            config, update_params, src="Global", dsts=["Det", "Cls", "Rec"]
        )
        return new_config

    def update_global_to_module(
        self, config, params: List[str], src: str, dsts: List[str]
    ):
        for dst in dsts:
            for param in params:
                config[dst].update({param: config[src][param]})
        return config

    def update_global_params(self, config, global_dict):
        if global_dict:
            config.update(global_dict)
        return config

    def update_params(
        self,
        config,
        param_dict: Dict[str, str],
        prefix: str,
        need_remove_prefix: Optional[List[str]] = None,
    ):
        if not param_dict:
            return config

        filter_dict = self.remove_prefix(param_dict, prefix, need_remove_prefix)
        model_path = filter_dict.get("model_path", None)
        if not model_path:
            filter_dict["model_path"] = str(root_dir / config["model_path"])

        config.update(filter_dict)
        return config

    @staticmethod
    def remove_prefix(
        config: Dict[str, str],
        prefix: str,
        need_remove_prefix: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        if not need_remove_prefix:
            return config

        new_rec_dict = {}
        for k, v in config.items():
            if k in need_remove_prefix:
                k = k.split(prefix)[1]
            new_rec_dict[k] = v
        return new_rec_dict
