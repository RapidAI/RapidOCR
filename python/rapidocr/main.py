# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import copy
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .cal_rec_boxes import CalRecBoxes
from .ch_ppocr_cls import TextClassifier, TextClsOutput
from .ch_ppocr_det import TextDetector, TextDetOutput
from .ch_ppocr_rec import TextRecInput, TextRecognizer, TextRecOutput
from .inference_engine.base import get_engine_name
from .utils import (
    LoadImage,
    ParseLang,
    RapidOCROutput,
    VisRes,
    add_round_letterbox,
    increase_min_side,
    reduce_max_side,
)
from .utils.parse_parameters import ParseParams

root_dir = Path(__file__).resolve().parent
DEFAULT_CFG_PATH = root_dir / "config.yaml"


class RapidOCR:
    def __init__(
        self, config_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None
    ):
        if config_path is not None and Path(config_path).exists():
            config = ParseParams.load(config_path)
        else:
            config = ParseParams.load(DEFAULT_CFG_PATH)

        if params:
            config = ParseParams.update_batch(config, params)

        engine_name = get_engine_name(config)

        lang_det, lang_rec = ParseLang()(config.Global.lang_det, config.Global.lang_rec)

        self.text_score = config.Global.text_score
        self.min_height = config.Global.min_height
        self.width_height_ratio = config.Global.width_height_ratio

        self.use_det = config.Global.use_det
        config.Det.lang = lang_det
        config.Det.engine_name = engine_name
        config.Det.engine_cfg = config.EngineConfig[engine_name]
        config.Det.task_type = "det"
        self.text_det = TextDetector(config.Det)

        self.use_cls = config.Global.use_cls
        config.Cls.lang = "ch_mobile"
        config.Cls.engine_name = engine_name
        config.Cls.engine_cfg = config.EngineConfig[engine_name]
        config.Cls.task_type = "cls"
        self.text_cls = TextClassifier(config.Cls)

        self.use_rec = config.Global.use_rec
        config.Rec.lang = lang_rec
        config.Rec.engine_name = engine_name
        config.Rec.engine_cfg = config.EngineConfig[engine_name]
        config.Rec.task_type = "rec"
        self.text_rec = TextRecognizer(config.Rec)

        self.load_img = LoadImage()
        self.max_side_len = config.Global.max_side_len
        self.min_side_len = config.Global.min_side_len

        self.cal_rec_boxes = CalRecBoxes()

        self.return_paddleocr_format = False
        self.return_word_box = False

        self.config = config

    def __call__(
        self,
        img_content: Union[str, np.ndarray, bytes, Path],
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
        return_word_box: bool = False,
        text_score: float = 0.5,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.6,
    ) -> RapidOCROutput:
        use_det = self.use_det if use_det is None else use_det
        use_cls = self.use_cls if use_cls is None else use_cls
        use_rec = self.use_rec if use_rec is None else use_rec

        self.return_word_box = return_word_box
        self.text_score = text_score
        self.text_det.postprocess_op.box_thresh = box_thresh
        self.text_det.postprocess_op.unclip_ratio = unclip_ratio

        ori_img = self.load_img(img_content)

        raw_h, raw_w = ori_img.shape[:2]
        op_record = {}
        img, ratio_h, ratio_w = self.preprocess(ori_img)
        op_record["preprocess"] = {"ratio_h": ratio_h, "ratio_w": ratio_w}

        det_res, cls_res, rec_res = TextDetOutput(), TextClsOutput(), TextRecOutput()

        if use_det:
            img, op_record = self.maybe_add_letterbox(img, op_record)
            det_res = self.text_det(img)
            if det_res.boxes is None:
                return RapidOCROutput()

            img = self.get_crop_img_list(img, det_res)

        if use_cls:
            cls_res = self.text_cls(img)
            img = cls_res.img_list

        if use_rec:
            rec_input = TextRecInput(img=img, return_word_box=self.return_word_box)
            rec_res = self.text_rec(rec_input)

        if (
            self.return_word_box
            and det_res.boxes is not None
            and all(v for v in rec_res.word_results)
        ):
            rec_res = self.cal_rec_boxes(img, det_res.boxes, rec_res)
            origin_words = []
            for one_word in rec_res.word_results:
                one_word_points = one_word[2]
                if one_word_points is None:
                    continue

                origin_words_points = self._get_origin_points(
                    [one_word_points], op_record, raw_h, raw_w
                )
                origin_words_points = origin_words_points.astype(np.int32).tolist()[0]
                origin_words.append((one_word[0], one_word[1], origin_words_points))
            rec_res.word_results = tuple(origin_words)

        if det_res.boxes is not None:
            det_res.boxes = self._get_origin_points(
                det_res.boxes, op_record, raw_h, raw_w
            )

        ocr_res = self.get_final_res(ori_img, det_res, cls_res, rec_res)
        return ocr_res

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        h, w = img.shape[:2]
        max_value = max(h, w)
        ratio_h = ratio_w = 1.0
        if max_value > self.max_side_len:
            img, ratio_h, ratio_w = reduce_max_side(img, self.max_side_len)

        h, w = img.shape[:2]
        min_value = min(h, w)
        if min_value < self.min_side_len:
            img, ratio_h, ratio_w = increase_min_side(img, self.min_side_len)
        return img, ratio_h, ratio_w

    def maybe_add_letterbox(
        self, img: np.ndarray, op_record: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = img.shape[:2]

        if self.width_height_ratio == -1:
            use_limit_ratio = False
        else:
            use_limit_ratio = w / h > self.width_height_ratio

        if h <= self.min_height or use_limit_ratio:
            padding_h = self._get_padding_h(h, w)
            block_img = add_round_letterbox(img, (padding_h, padding_h, 0, 0))
            op_record["padding_1"] = {"top": padding_h, "left": 0}
            return block_img, op_record

        op_record["padding_1"] = {"top": 0, "left": 0}
        return img, op_record

    def _get_padding_h(self, h: int, w: int) -> int:
        new_h = max(int(w / self.width_height_ratio), self.min_height) * 2
        padding_h = int(abs(new_h - h) / 2)
        return padding_h

    def get_crop_img_list(
        self, img: np.ndarray, det_res: TextDetOutput
    ) -> List[np.ndarray]:
        def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
            img_crop_width = int(
                max(
                    np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[2] - points[3]),
                )
            )
            img_crop_height = int(
                max(
                    np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2]),
                )
            )
            pts_std = np.array(
                [
                    [0, 0],
                    [img_crop_width, 0],
                    [img_crop_width, img_crop_height],
                    [0, img_crop_height],
                ]
            ).astype(np.float32)
            M = cv2.getPerspectiveTransform(points, pts_std)
            dst_img = cv2.warpPerspective(
                img,
                M,
                (img_crop_width, img_crop_height),
                borderMode=cv2.BORDER_REPLICATE,
                flags=cv2.INTER_CUBIC,
            )
            dst_img_height, dst_img_width = dst_img.shape[0:2]
            if dst_img_height * 1.0 / dst_img_width >= 1.5:
                dst_img = np.rot90(dst_img)
            return dst_img

        img_crop_list = []
        for box in det_res.boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        return img_crop_list

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
        )
        ocr_res = self.filter_by_text_score(ocr_res)
        if len(ocr_res) <= 0:
            return RapidOCROutput()

        if self.return_paddleocr_format:
            return ocr_res.to_paddleocr_format()

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

    def export_config(self, save_path: Union[Path, str]) -> None:
        with open(save_path, "w", encoding="utf-8") as f:
            ParseParams.save(self.config, f)


def generate_cfg(args):
    if args.save_cfg_file is None:
        args.save_cfg_file = "./default_rapidocr.yaml"

    shutil.copyfile(DEFAULT_CFG_PATH, args.save_cfg_file)
    print(f"The config file has saved in {args.save_cfg_file}")


def check_install():
    img_url = "https://github.com/RapidAI/RapidOCR/blob/a9bb7c1f44b6e00556ada90ac588f020d7637c4b/python/tests/test_files/ch_en_num.jpg?raw=true"
    result = RapidOCR()(img_url)

    if result.txts is None or result.txts[0] != "正品促销":
        raise ValueError("The installation is incorrect!")

    print("Success! rapidocr is installed correctly!")


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

    if args.command == "check":
        check_install()
        return

    params = {
        "Global.text_score": args.text_score,
        "Global.return_word_box": args.return_word_box,
    }
    ocr_engine = RapidOCR(params=params)

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
            words_results = result.word_results
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
