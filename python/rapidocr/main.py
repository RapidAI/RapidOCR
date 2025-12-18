# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from omegaconf import DictConfig

from .cal_rec_boxes import CalRecBoxes
from .ch_ppocr_cls import TextClassifier, TextClsOutput
from .ch_ppocr_det import TextDetector, TextDetOutput
from .ch_ppocr_rec import TextRecInput, TextRecognizer, TextRecOutput
from .cli import check_install, generate_cfg
from .utils.load_image import LoadImage
from .utils.log import logger
from .utils.output import RapidOCROutput
from .utils.parse_parameters import ParseParams
from .utils.process_img import (
    apply_vertical_padding,
    get_rotate_crop_image,
    map_boxes_to_original,
    map_img_to_original,
    resize_image_within_bounds,
)
from .utils.typings import LangRec
from .utils.utils import filter_by_indices
from .utils.vis_res import VisRes

root_dir = Path(__file__).resolve().parent
DEFAULT_CFG_PATH = root_dir / "config.yaml"


class RapidOCR:
    def __init__(
        self, config_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None
    ):
        cfg = self._load_config(config_path, params)

        logger.setLevel(cfg.Global.log_level.upper())

        self._initialize(cfg)

    def _load_config(
        self, config_path: Optional[str], params: Optional[Dict[str, Any]]
    ) -> DictConfig:
        if config_path is not None and Path(config_path).exists():
            cfg = ParseParams.load(config_path)
        else:
            cfg = ParseParams.load(DEFAULT_CFG_PATH)

        if params:
            cfg = ParseParams.update_batch(cfg, params)
        return cfg

    def _initialize(self, cfg: DictConfig):
        self.text_score = cfg.Global.text_score
        self.min_height = cfg.Global.min_height
        self.width_height_ratio = cfg.Global.width_height_ratio

        self.use_det = cfg.Global.use_det
        cfg.Det.engine_cfg = cfg.EngineConfig[cfg.Det.engine_type.value]
        self.text_det = TextDetector(cfg.Det)

        self.use_cls = cfg.Global.use_cls
        cfg.Cls.engine_cfg = cfg.EngineConfig[cfg.Cls.engine_type.value]
        self.text_cls = TextClassifier(cfg.Cls)

        self.use_rec = cfg.Global.use_rec
        cfg.Rec.engine_cfg = cfg.EngineConfig[cfg.Rec.engine_type.value]
        cfg.Rec.font_path = cfg.Global.font_path
        self.text_rec = TextRecognizer(cfg.Rec)

        self.load_img = LoadImage()
        self.max_side_len = cfg.Global.max_side_len
        self.min_side_len = cfg.Global.min_side_len

        self.cal_rec_boxes = CalRecBoxes()

        self.return_word_box = cfg.Global.return_word_box
        self.return_single_char_box = cfg.Global.return_single_char_box

        self.cfg = cfg

    def __call__(
        self,
        img_content: Union[str, np.ndarray, bytes, Path],
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
        return_word_box: Optional[bool] = None,
        return_single_char_box: Optional[bool] = None,
        text_score: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
    ) -> Union[TextDetOutput, TextClsOutput, TextRecOutput, RapidOCROutput]:
        self.update_params(
            use_det=use_det,
            use_cls=use_cls,
            use_rec=use_rec,
            return_word_box=return_word_box,
            return_single_char_box=return_single_char_box,
            text_score=text_score,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
        )

        ori_img = self.load_img(img_content)
        img, op_record = self.preprocess_img(ori_img)
        det_res, cls_res, rec_res, cropped_img_list = self.run_ocr_steps(img, op_record)
        return self.build_final_output(
            ori_img, det_res, cls_res, rec_res, cropped_img_list, op_record
        )

    def run_ocr_steps(self, img: np.ndarray, op_record: Dict[str, Any]):
        det_res, cls_res, rec_res = TextDetOutput(), TextClsOutput(), TextRecOutput()

        if self.use_det:
            try:
                cropped_img_list, det_res = self.detect_and_crop(img, op_record)
            except RapidOCRError as e:
                logger.warning(e)
                return TextDetOutput(), TextClsOutput(), TextRecOutput(), []
        else:
            cropped_img_list = [img]

        if self.use_cls:
            try:
                cls_img_list, cls_res = self.cls_and_rotate(cropped_img_list)
            except RapidOCRError as e:
                logger.warning(e)
                return det_res, TextClsOutput(), TextRecOutput(), []
        else:
            cls_img_list = cropped_img_list

        if self.use_rec:
            try:
                rec_res = self.recognize_txt(cls_img_list)
            except RapidOCRError as e:
                logger.warning(e)
                return det_res, cls_res, TextRecOutput(), []

        return det_res, cls_res, rec_res, cropped_img_list

    def build_final_output(
        self,
        ori_img: np.ndarray,
        det_res: TextDetOutput,
        cls_res: TextClsOutput,
        rec_res: TextRecOutput,
        cropped_img_list: List[np.ndarray],
        op_record: Dict[str, Any],
    ) -> Union[TextDetOutput, TextClsOutput, TextRecOutput, RapidOCROutput]:
        ori_h, ori_w = ori_img.shape[:2]

        if det_res.boxes is not None:
            det_res.boxes = map_boxes_to_original(
                det_res.boxes, op_record, ori_h, ori_w
            )

            ratio_h = op_record["preprocess"]["ratio_h"]
            ratio_w = op_record["preprocess"]["ratio_w"]
            cropped_img_list = map_img_to_original(cropped_img_list, ratio_h, ratio_w)

        # 过滤识别结果为空的值
        if (
            rec_res.txts is not None
            and det_res.boxes is not None
            and det_res.scores is not None
        ):
            valid_ids = [i for i, v in enumerate(rec_res.txts) if v.strip()]

            det_res.boxes = filter_by_indices(det_res.boxes, valid_ids)
            det_res.scores = filter_by_indices(det_res.scores, valid_ids)

            rec_res.imgs = filter_by_indices(rec_res.imgs, valid_ids)
            rec_res.txts = filter_by_indices(rec_res.txts, valid_ids)
            rec_res.word_results = filter_by_indices(rec_res.word_results, valid_ids)
            rec_res.scores = filter_by_indices(rec_res.scores, valid_ids)

            cropped_img_list = filter_by_indices(cropped_img_list, valid_ids)

        # 仅分类结果
        if (
            det_res.boxes is None
            and rec_res.txts is None
            and cls_res.cls_res is not None
        ):
            return cls_res

        # 无有效输出
        if det_res.boxes is None and rec_res.txts is None:
            return RapidOCROutput()

        # 仅识别结果（无检测）
        if det_res.boxes is None and rec_res.txts is not None:
            return rec_res

        # 仅检测结果（无识别）
        if det_res.boxes is not None and rec_res.txts is None:
            return det_res

        if (
            self.return_word_box
            and det_res.boxes is not None
            and all(rec_res.word_results)
        ):
            rec_res.word_results = self.calc_word_boxes(
                cropped_img_list, det_res.boxes, rec_res, op_record, ori_h, ori_w
            )

        ocr_res = RapidOCROutput(
            img=ori_img,
            boxes=det_res.boxes,
            txts=rec_res.txts,
            scores=rec_res.scores,
            word_results=rec_res.word_results,
            elapse_list=[det_res.elapse, cls_res.elapse, rec_res.elapse],
            viser=VisRes(
                text_score=self.cfg.Global.text_score,
                lang_type=self.cfg.Rec.lang_type,
                font_path=self.cfg.Global.font_path,
            ),
        )

        ocr_res = self.filter_by_text_score(ocr_res)
        return ocr_res if len(ocr_res) > 0 else RapidOCROutput()

    def calc_word_boxes(
        self,
        img: List[np.ndarray],
        dt_boxes: np.ndarray,
        rec_res: TextRecOutput,
        op_record: Dict[str, Any],
        raw_h: int,
        raw_w: int,
    ) -> Any:
        rec_res = self.cal_rec_boxes(
            img, dt_boxes, rec_res, self.return_single_char_box
        )

        origin_words = []
        for word_line in rec_res.word_results:
            origin_words_item = [
                (txt, score, bbox) for txt, score, bbox in word_line if bbox is not None
            ]
            if origin_words_item:
                origin_words.append(tuple(origin_words_item))
        return tuple(origin_words)

    def update_params(self, **kwargs):
        param_map = {
            "use_det": ("use_det",),
            "use_cls": ("use_cls",),
            "use_rec": ("use_rec",),
            "return_word_box": ("return_word_box",),
            "return_single_char_box": ("return_single_char_box",),
            "text_score": ("text_score",),
            "box_thresh": ("text_det", "postprocess_op", "box_thresh"),
            "unclip_ratio": ("text_det", "postprocess_op", "unclip_ratio"),
        }

        for key, value in kwargs.items():
            if value is None:
                continue

            path = param_map.get(key)
            if not path:
                raise ValueError(f"Unknown parameter: {key}")

            obj = self
            for attr in path[:-1]:
                obj = getattr(obj, attr)

            setattr(obj, path[-1], value)

    def preprocess_img(self, ori_img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        op_record = {}
        img, ratio_h, ratio_w = resize_image_within_bounds(
            ori_img, self.min_side_len, self.max_side_len
        )
        op_record["preprocess"] = {"ratio_h": ratio_h, "ratio_w": ratio_w}
        return img, op_record

    def detect_and_crop(
        self, img: np.ndarray, op_record: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], TextDetOutput]:
        img, op_record = apply_vertical_padding(
            img, op_record, self.width_height_ratio, self.min_height
        )
        det_res = self.text_det(img)

        if det_res.boxes is None:
            raise RapidOCRError("The text detection result is empty")

        img_crop_list = self.crop_text_regions(img, det_res.boxes)
        return img_crop_list, det_res

    def crop_text_regions(
        self, img: np.ndarray, det_boxes: np.ndarray
    ) -> List[np.ndarray]:
        img_crop_list = []
        for box in det_boxes:
            img_crop = get_rotate_crop_image(img, copy.deepcopy(box))
            img_crop_list.append(img_crop)
        return img_crop_list

    def cls_and_rotate(
        self, img: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], TextClsOutput]:
        cls_res = self.text_cls(img)
        if cls_res.img_list is None:
            raise RapidOCRError("The text classifier is empty")
        return cls_res.img_list, cls_res

    def recognize_txt(self, img: List[np.ndarray]) -> TextRecOutput:
        rec_input = TextRecInput(img=img, return_word_box=self.return_word_box)

        rec_res = self.text_rec(rec_input)
        if rec_res.txts is None:
            raise RapidOCRError("The text recognize result is empty")

        return rec_res

    def filter_by_text_score(self, ocr_res: RapidOCROutput) -> RapidOCROutput:
        filter_boxes, filter_txts, filter_scores, filter_words = [], [], [], []
        for i, (box, txt, score) in enumerate(
            zip(ocr_res.boxes, ocr_res.txts, ocr_res.scores)
        ):
            if score < self.text_score:
                continue

            if ocr_res.word_results[i]:
                filter_words.append(ocr_res.word_results[i])

            filter_boxes.append(box)
            filter_txts.append(txt)
            filter_scores.append(score)

        ocr_res.boxes = np.array(filter_boxes)
        ocr_res.txts = tuple(filter_txts)
        ocr_res.scores = tuple(filter_scores)
        ocr_res.word_results = tuple(filter_words)
        return ocr_res


class RapidOCRError(Exception):
    pass


def parse_args(arg_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", "--img_path", type=str, default=None)
    parser.add_argument("--text_score", type=float, default=0.5)
    parser.add_argument(
        "--lang_type",
        type=str,
        default="ch",
        choices=list(v.value for v in LangRec),
    )
    parser.add_argument("-vis", "--vis_res", action="store_true", default=False)
    parser.add_argument("--vis_save_dir", type=Path, default=".")
    parser.add_argument("--font_path", type=str, default=None)
    parser.add_argument(
        "-word", "--return_word_box", action="store_true", default=False
    )

    subparser = parser.add_subparsers(dest="command", help="Sub-command help")
    parser_cfg = subparser.add_parser("config", help="Generate config file")
    parser_cfg.add_argument("--save_cfg_file", type=Path, default=None)
    parser_cfg.set_defaults(func=generate_cfg)

    parser_check = subparser.add_parser(
        "check", help="Check if it is installed correctly "
    )
    parser_check.set_defaults(func=check_install)

    args = parser.parse_args(arg_list)
    return args


def main(arg_list: Optional[List[str]] = None):
    args = parse_args(arg_list)

    if args.command == "config":
        generate_cfg(args)
        return

    params = {
        "Global.text_score": args.text_score,
        "Global.return_word_box": args.return_word_box,
    }
    ocr_engine = RapidOCR(params=params)

    if args.command == "check":
        check_install(ocr_engine)
        return

    if args.img_path is None:
        raise ValueError("Please input the image path")

    if args.return_word_box:
        result = ocr_engine(args.img_path, return_word_box=args.return_word_box)
    else:
        result = ocr_engine(args.img_path)

    print(result)

    if args.vis_res:
        vis = VisRes(
            text_score=args.text_score,
            font_path=args.font_path,
            lang_type=LangRec(args.lang_type),
        )
        cur_dir = args.vis_save_dir

        if args.return_word_box:
            words_results = sum(result.word_results, ())
            words, words_scores, words_boxes = list(zip(*words_results))
            vis_img = vis(args.img_path, words_boxes, words, words_scores)
            save_path = cur_dir / f"{Path(args.img_path).stem}_vis_single.png"
            cv2.imwrite(str(save_path), vis_img)
            print(f"The vis single result has saved in {save_path}")
            return

        save_path = cur_dir / f"{Path(args.img_path).stem}_vis.png"
        vis_img = vis(args.img_path, result.boxes, result.txts, result.scores)
        cv2.imwrite(str(save_path), vis_img)
        print(f"The vis result has saved in {save_path}")


if __name__ == "__main__":
    main()
