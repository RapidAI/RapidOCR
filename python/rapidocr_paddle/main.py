# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import copy
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .ch_ppocr_v2_cls import TextClassifier
from .ch_ppocr_v3_det import TextDetector
from .ch_ppocr_v3_rec import TextRecognizer
from .utils import (
    LoadImage,
    UpdateParameters,
    VisRes,
    concat_model_path,
    init_args,
    read_yaml,
)

root_dir = Path(__file__).resolve().parent


class RapidOCR:
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        if config_path is None:
            config_path = str(root_dir / "config.yaml")

        if not Path(config_path).exists():
            raise FileExistsError(f"{config_path} does not exist!")

        config = read_yaml(config_path)
        config = concat_model_path(config)

        if kwargs:
            updater = UpdateParameters()
            config = updater(config, **kwargs)

        global_config = config["Global"]
        self.print_verbose = global_config["print_verbose"]
        self.text_score = global_config["text_score"]
        self.min_height = global_config["min_height"]
        self.width_height_ratio = global_config["width_height_ratio"]

        self.use_det = global_config["use_det"]
        self.text_det = TextDetector(config["Det"])

        self.use_cls = global_config["use_cls"]
        self.text_cls = TextClassifier(config["Cls"])

        self.use_rec = global_config["use_rec"]
        self.text_rec = TextRecognizer(config["Rec"])

        self.load_img = LoadImage()

    def __call__(
        self,
        img_content: Union[str, np.ndarray, bytes, Path],
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
        **kwargs,
    ):
        use_det = self.use_det if use_det is None else use_det
        use_cls = self.use_cls if use_cls is None else use_cls
        use_rec = self.use_rec if use_rec is None else use_rec

        if kwargs:
            box_thresh = kwargs.get("box_thresh", 0.5)
            unclip_ratio = kwargs.get("unclip_ratio", 1.6)
            text_score = kwargs.get("text_score", 0.5)

            self.text_det.postprocess_op.box_thresh = box_thresh
            self.text_det.postprocess_op.unclip_ratio = unclip_ratio
            self.text_score = text_score

        img = self.load_img(img_content)

        dt_boxes, cls_res, rec_res = None, None, None
        det_elapse, cls_elapse, rec_elapse = 0.0, 0.0, 0.0

        if use_det:
            dt_boxes, det_elapse = self.auto_text_det(img)
            if dt_boxes is None:
                return None, None

            img = self.get_crop_img_list(img, dt_boxes)

        if use_cls:
            img, cls_res, cls_elapse = self.text_cls(img)

        if use_rec:
            rec_res, rec_elapse = self.text_rec(img)

        ocr_res = self.get_final_res(
            dt_boxes, cls_res, rec_res, det_elapse, cls_elapse, rec_elapse
        )
        return ocr_res

    def auto_text_det(
        self,
        img: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float, Optional[List[np.ndarray]]]:
        h, w = img.shape[:2]
        if self.width_height_ratio == -1:
            use_limit_ratio = False
        else:
            use_limit_ratio = w / h > self.width_height_ratio

        if h <= self.min_height or use_limit_ratio:
            logging.warning(
                "Because the aspect ratio of the current image exceeds the limit (min_height or width_height_ratio), the program will skip the detection step."
            )
            dt_boxes = self.get_boxes_img_without_det(h, w)
            return dt_boxes, 0.0

        dt_boxes, det_elapse = self.text_det(img)
        if dt_boxes is None or len(dt_boxes) < 1:
            return None, 0.0

        dt_boxes = self.sorted_boxes(dt_boxes)
        return dt_boxes, det_elapse

    def get_boxes_img_without_det(self, h, w):
        x0, y0, x1, y1 = 0, 0, w, h
        dt_boxes = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        dt_boxes = dt_boxes[np.newaxis, ...]
        return dt_boxes

    def get_crop_img_list(self, img, dt_boxes):
        def get_rotate_crop_image(img, points):
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
            pts_std = np.float32(
                [
                    [0, 0],
                    [img_crop_width, 0],
                    [img_crop_width, img_crop_height],
                    [0, img_crop_height],
                ]
            )
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
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        return img_crop_list

    @staticmethod
    def sorted_boxes(dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if (
                    abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10
                    and _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def get_final_res(
        self, dt_boxes, cls_res, rec_res, det_elapse, cls_elapse, rec_elapse
    ):
        if dt_boxes is None and rec_res is None and cls_res is not None:
            return cls_res, [cls_elapse]

        if dt_boxes is None and rec_res is None:
            return None, None

        if dt_boxes is None and rec_res is not None:
            return [[res[0], res[1]] for res in rec_res], [rec_elapse]

        if dt_boxes is not None and rec_res is None:
            return [box.tolist() for box in dt_boxes], [det_elapse]

        dt_boxes, rec_res = self.filter_result(dt_boxes, rec_res)
        if len(dt_boxes) <= 0:
            return None, None

        ocr_res = [
            [box.tolist(), res[0], res[1]] for box, res in zip(dt_boxes, rec_res)
        ], [det_elapse, cls_elapse, rec_elapse]
        return ocr_res

    def filter_result(self, dt_boxes, rec_res):
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if float(score) >= self.text_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)

        return filter_boxes, filter_rec_res


def main():
    args = init_args()
    ocr_engine = RapidOCR(**vars(args))

    use_det = not args.no_det
    use_cls = not args.no_cls
    use_rec = not args.no_rec
    result, elapse_list = ocr_engine(
        args.img_path, use_det=use_det, use_cls=use_cls, use_rec=use_rec
    )
    print(result)

    if args.print_cost:
        print(elapse_list)

    if args.vis_res:
        vis = VisRes()
        Path(args.vis_save_path).mkdir(parents=True, exist_ok=True)
        save_path = Path(args.vis_save_path) / f"{Path(args.img_path).stem}_vis.png"

        if use_det and not use_cls and not use_rec:
            boxes, *_ = list(zip(*result))
            vis_img = vis(args.img_path, boxes)
            cv2.imwrite(str(save_path), vis_img)
            print(f"The vis result has saved in {save_path}")

        elif use_det and use_rec:
            font_path = Path(args.vis_font_path)
            if not font_path.exists():
                raise FileExistsError(f"{font_path} does not exist!")

            boxes, txts, scores = list(zip(*result))
            vis_img = vis(args.img_path, boxes, txts, scores, font_path=font_path)
            cv2.imwrite(str(save_path), vis_img)
            print(f"The vis result has saved in {save_path}")


if __name__ == "__main__":
    main()
