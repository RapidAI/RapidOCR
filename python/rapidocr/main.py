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
from .utils import (
    LoadImage,
    Logger,
    RapidOCROutput,
    VisRes,
    add_round_letterbox,
    get_padding_h,
    get_rotate_crop_image,
    resize_image_within_bounds,
)
from .utils.parse_parameters import ParseParams

root_dir = Path(__file__).resolve().parent
DEFAULT_CFG_PATH = root_dir / "config.yaml"


class RapidOCR:
    def __init__(
        self, config_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None
    ):
        cfg = self.load_config(config_path, params)
        self.initialize(cfg)

        self.logger = Logger(logger_name=__name__).get_log()

    def load_config(
        self, config_path: Optional[str], params: Optional[Dict[str, Any]]
    ) -> DictConfig:
        if config_path is not None and Path(config_path).exists():
            cfg = ParseParams.load(config_path)
        else:
            cfg = ParseParams.load(DEFAULT_CFG_PATH)

        if params:
            cfg = ParseParams.update_batch(cfg, params)
        return cfg

    def initialize(self, cfg: DictConfig):
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
        return_word_box: bool = False,
        return_single_char_box: bool = False,
        text_score: float = 0.5,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.6,
    ) -> Union[TextDetOutput, TextClsOutput, TextRecOutput, RapidOCROutput]:
        self.update_params(
            use_det,
            use_cls,
            use_rec,
            return_word_box,
            return_single_char_box,
            text_score,
            box_thresh,
            unclip_ratio,
        )

        ori_img = self.load_img(img_content)
        img, op_record = self.preprocess_img(ori_img)

        det_res, cls_res, rec_res = TextDetOutput(), TextClsOutput(), TextRecOutput()
        if self.use_det:
            try:
                img, det_res = self.get_det_res(img, op_record)
            except RapidOCRError as e:
                self.logger.warning(e)
                return RapidOCROutput()

        if self.use_cls:
            try:
                img, cls_res = self.get_cls_res(img)
            except RapidOCRError as e:
                self.logger.warning(e)
                return RapidOCROutput()

        if self.use_rec:
            rec_res = self.get_rec_res(img)

        return self.finalize_results(ori_img, det_res, cls_res, rec_res, img, op_record)

    def finalize_results(
        self,
        ori_img: np.ndarray,
        det_res: TextDetOutput,
        cls_res: TextClsOutput,
        rec_res: TextRecOutput,
        img: List[np.ndarray],
        op_record: Dict[str, Any],
    ) -> Union[TextDetOutput, TextClsOutput, TextRecOutput, RapidOCROutput]:
        raw_h, raw_w = ori_img.shape[:2]

        if (
            self.return_word_box
            and det_res.boxes is not None
            and all(v for v in rec_res.word_results)
        ):
            rec_res.word_results = self.calc_word_boxes(
                img, det_res, rec_res, op_record, raw_h, raw_w
            )

        if det_res.boxes is not None:
            det_res.boxes = self._get_origin_points(
                det_res.boxes, op_record, raw_h, raw_w
            )
        return self.get_final_res(ori_img, det_res, cls_res, rec_res)

    def calc_word_boxes(
        self,
        img: List[np.ndarray],
        det_res: TextDetOutput,
        rec_res: TextRecOutput,
        op_record: Dict[str, Any],
        raw_h: int,
        raw_w: int,
    ) -> Any:
        rec_res = self.cal_rec_boxes(
            img, det_res.boxes, rec_res, self.return_single_char_box
        )

        origin_words = []
        for word_line in rec_res.word_results:
            origin_words_item = []
            for txt, score, bbox in word_line:
                if bbox is None:
                    continue

                origin_words_points = self._get_origin_points(
                    [bbox], op_record, raw_h, raw_w
                )
                origin_words_points = origin_words_points.astype(np.int32).tolist()[0]
                origin_words_item.append((txt, score, origin_words_points))

            if origin_words_item:
                origin_words.append(tuple(origin_words_item))
        return tuple(origin_words)

    def update_params(
        self,
        use_det,
        use_cls,
        use_rec,
        return_word_box,
        return_single_char_box,
        text_score,
        box_thresh,
        unclip_ratio,
    ):
        self.use_det = self.use_det if use_det is None else use_det
        self.use_cls = self.use_cls if use_cls is None else use_cls
        self.use_rec = self.use_rec if use_rec is None else use_rec

        self.return_word_box = return_word_box
        self.return_single_char_box = return_single_char_box
        self.text_score = text_score
        self.text_det.postprocess_op.box_thresh = box_thresh
        self.text_det.postprocess_op.unclip_ratio = unclip_ratio

    def preprocess_img(self, ori_img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        op_record = {}
        img, ratio_h, ratio_w = resize_image_within_bounds(
            ori_img, self.min_side_len, self.max_side_len
        )
        op_record["preprocess"] = {"ratio_h": ratio_h, "ratio_w": ratio_w}
        return img, op_record

    def get_det_res(
        self, img: np.ndarray, op_record: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], TextDetOutput]:
        img, op_record = self._add_letterbox(img, op_record)
        det_res = self.text_det(img)
        if det_res.boxes is None:
            raise RapidOCRError("The text detection result is empty")

        img_list = self.get_crop_img_list(img, det_res)
        return img_list, det_res

    def get_crop_img_list(
        self, img: np.ndarray, det_res: TextDetOutput
    ) -> List[np.ndarray]:
        img_crop_list = []
        for box in det_res.boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        return img_crop_list

    def get_cls_res(
        self, img: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], TextClsOutput]:
        cls_res = self.text_cls(img)
        if cls_res.img_list is None:
            raise RapidOCRError("The text classifier is empty")
        return cls_res.img_list, cls_res

    def get_rec_res(self, img: List[np.ndarray]) -> TextRecOutput:
        rec_input = TextRecInput(img=img, return_word_box=self.return_word_box)
        return self.text_rec(rec_input)

    def get_final_res(
        self,
        ori_img: np.ndarray,
        det_res: TextDetOutput,
        cls_res: TextClsOutput,
        rec_res: TextRecOutput,
    ) -> Union[TextDetOutput, TextClsOutput, TextRecOutput, RapidOCROutput]:
        dt_boxes = det_res.boxes
        txt_res = rec_res.txts

        if dt_boxes is None and txt_res is None and cls_res.cls_res is not None:
            return cls_res

        if dt_boxes is None and txt_res is None:
            return RapidOCROutput()

        if dt_boxes is None and txt_res is not None:
            return rec_res

        if dt_boxes is not None and txt_res is None:
            return det_res

        ocr_res = RapidOCROutput(
            img=ori_img,
            boxes=det_res.boxes,
            txts=rec_res.txts,
            scores=rec_res.scores,
            word_results=rec_res.word_results,
            elapse_list=[det_res.elapse, cls_res.elapse, rec_res.elapse],
            lang_type=self.cfg.Rec.lang_type,
        )
        ocr_res = self.filter_by_text_score(ocr_res)
        if len(ocr_res) <= 0:
            return RapidOCROutput()

        return ocr_res

    def filter_by_text_score(self, ocr_res: RapidOCROutput) -> RapidOCROutput:
        filter_boxes, filter_txts, filter_scores = [], [], []
        for box, txt, score in zip(ocr_res.boxes, ocr_res.txts, ocr_res.scores):
            if float(score) >= self.text_score:
                filter_boxes.append(box)
                filter_txts.append(txt)
                filter_scores.append(score)

        ocr_res.boxes = np.array(filter_boxes)
        ocr_res.txts = tuple(filter_txts)
        ocr_res.scores = tuple(filter_scores)
        return ocr_res

    def _add_letterbox(
        self, img: np.ndarray, op_record: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = img.shape[:2]

        if self.width_height_ratio == -1:
            use_limit_ratio = False
        else:
            use_limit_ratio = w / h > self.width_height_ratio

        if h <= self.min_height or use_limit_ratio:
            padding_h = get_padding_h(h, w, self.width_height_ratio, self.min_height)
            block_img = add_round_letterbox(img, (padding_h, padding_h, 0, 0))
            op_record["padding_1"] = {"top": padding_h, "left": 0}
            return block_img, op_record

        op_record["padding_1"] = {"top": 0, "left": 0}
        return img, op_record

    def _get_origin_points(
        self,
        dt_boxes: List[np.ndarray],
        op_record: Dict[str, Any],
        raw_h: int,
        raw_w: int,
    ) -> np.ndarray:
        dt_boxes_array = np.array(dt_boxes).astype(np.float32)
        for op in reversed(list(op_record.keys())):
            v = op_record[op]
            if "padding" in op:
                top, left = v.get("top"), v.get("left")
                dt_boxes_array[:, :, 0] -= left
                dt_boxes_array[:, :, 1] -= top
            elif "preprocess" in op:
                ratio_h = v.get("ratio_h")
                ratio_w = v.get("ratio_w")
                dt_boxes_array[:, :, 0] *= ratio_w
                dt_boxes_array[:, :, 1] *= ratio_h

        dt_boxes_array = np.where(dt_boxes_array < 0, 0, dt_boxes_array)
        dt_boxes_array[..., 0] = np.where(
            dt_boxes_array[..., 0] > raw_w, raw_w, dt_boxes_array[..., 0]
        )
        dt_boxes_array[..., 1] = np.where(
            dt_boxes_array[..., 1] > raw_h, raw_h, dt_boxes_array[..., 1]
        )
        return dt_boxes_array


class RapidOCRError(Exception):
    pass


def parse_args(arg_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", "--img_path", type=str, default=None)
    parser.add_argument("--text_score", type=float, default=0.5)
    parser.add_argument("-vis", "--vis_res", action="store_true", default=False)
    parser.add_argument("--vis_save_dir", type=Path, default=".")
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
        vis = VisRes()
        cur_dir = args.vis_save_dir

        if args.return_word_box:
            words_results = sum(result.word_results, ())
            words, words_scores, words_boxes = list(zip(*words_results))
            vis_img = vis(args.img_path, words_boxes, words, words_scores)
            save_path = cur_dir / f"{Path(args.img_path).stem}_vis_single.png"
            cv2.imwrite(str(save_path), vis_img)
            print(f"The vis single result has saved in {save_path}")
        else:
            save_path = cur_dir / f"{Path(args.img_path).stem}_vis.png"
            vis_img = vis(args.img_path, result.boxes, result.txts, result.scores)
            cv2.imwrite(str(save_path), vis_img)
            print(f"The vis result has saved in {save_path}")


if __name__ == "__main__":
    main()
