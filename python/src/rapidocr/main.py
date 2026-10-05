"""RapidOCR entry point backed by the native PP-OCRv6 engine.

The constructor and ``__call__`` follow ``rapidocr.main.RapidOCR`` on the
``main`` branch. This build always runs the in-process C++ ONNX interpreter
(detector + recognizer). Angle classification and word-box decoding are not
part of that engine; ``use_cls`` and ``return_word_box`` are accepted and
reported rather than silently changing results.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from . import _native
from .download import download_models
from .output import RapidOCROutput, filter_by_score
from .typings import EngineType, OCRVersion
from .utils.load_image import LoadImage, LoadImageError
from .utils.parse_parameters import ConfigNode, ParseParams
from .utils.process_img import apply_vertical_padding, map_boxes_to_original, resize_image_within_bounds

logger = logging.getLogger("rapidocr")

_NATIVE_ENGINES = {EngineType.ONNXRUNTIME.value, EngineType.PPOCR_CPP.value, "onnxruntime", "ppocr_cpp"}


class RapidOCRError(Exception):
    pass


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


class RapidOCR:
    def __init__(self, config_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        self.cfg = _load_config(config_path, params)
        level = str(self.cfg.Global.get("log_level", "info")).upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        self.text_score = float(self.cfg.Global.text_score)
        self.use_det = bool(self.cfg.Global.use_det)
        self.use_cls = bool(self.cfg.Global.use_cls)
        self.use_rec = bool(self.cfg.Global.use_rec)
        self.return_word_box = bool(self.cfg.Global.get("return_word_box", False))
        self.return_single_char_box = bool(self.cfg.Global.get("return_single_char_box", False))
        self.min_side_len = self.cfg.Global.get("min_side_len", 30)
        self.max_side_len = self.cfg.Global.get("max_side_len", 2000)
        self.min_height = self.cfg.Global.get("min_height", 30)
        self.width_height_ratio = self.cfg.Global.get("width_height_ratio", 8)
        self.text_det = None
        self.text_cls = None
        self.text_rec = None
        self.load_img = LoadImage()
        self._engine: Optional[_native.NativeOCR] = None
        self._warned_cls = False
        self._check_engine_choice()

    def update_params(
        self,
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
        return_word_box: Optional[bool] = None,
        return_single_char_box: Optional[bool] = None,
        text_score: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        values = {
            "use_det": use_det,
            "use_cls": use_cls,
            "use_rec": use_rec,
            "return_word_box": return_word_box,
            "return_single_char_box": return_single_char_box,
            "text_score": text_score,
            "box_thresh": box_thresh,
            "unclip_ratio": unclip_ratio,
        }
        values.update(kwargs)
        for key, value in values.items():
            if value is None:
                continue
            if key in {"box_thresh", "unclip_ratio"}:
                setattr(self.cfg.Det, key, value)
                if self.text_det is not None:
                    setattr(self.text_det.postprocess_op, key, value)
                continue
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
            if key in {"use_det", "use_cls", "use_rec", "return_word_box", "return_single_char_box", "text_score"}:
                setattr(self.cfg.Global, key if key != "text_score" else "text_score", value)

    def __call__(
        self,
        img_content: Any,
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
        return_word_box: Optional[bool] = None,
        return_single_char_box: Optional[bool] = None,
        text_score: Optional[float] = None,
        box_thresh: Optional[float] = None,
        unclip_ratio: Optional[float] = None,
    ) -> RapidOCROutput:
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
        if return_word_box:
            logger.warning("return_word_box is not implemented by the native PP-OCRv6 engine")
        if return_single_char_box:
            logger.warning("return_single_char_box is not implemented by the native PP-OCRv6 engine")
        if not self.use_det or not self.use_rec:
            raise RapidOCRError(
                "The native engine runs detection and recognition together. "
                "use_det=False or use_rec=False is not supported."
            )
        if self.use_cls and not self._warned_cls:
            logger.warning("use_cls is ignored: the native engine has no angle classifier")
            self._warned_cls = True

        started = time.perf_counter()
        original = self.load_img(img_content)
        prepared, ratio_w, ratio_h, pad_top = _prepare_image(self, original)
        engine = self._ensure_engine()
        det_threshold = float(self.cfg.Det.get("thresh", 0.3))
        box_threshold = float(self.cfg.Det.get("box_thresh", 0.5))
        unclip = float(self.cfg.Det.get("unclip_ratio", 1.6))
        engine.set_det_thresholds(det_threshold, box_threshold, unclip)
        boxes, txts, scores = engine.recognize(prepared)
        kept_boxes, kept_txts, kept_scores = filter_by_score(boxes, txts, scores, self.text_score)
        elapsed = time.perf_counter() - started
        if len(kept_txts) == 0:
            return RapidOCROutput()
        ori_h, ori_w = original.shape[:2]
        kept_boxes = map_boxes_to_original(kept_boxes, ratio_w, ratio_h, pad_top, ori_w, ori_h)
        return RapidOCROutput(
            img=original,
            boxes=kept_boxes,
            txts=kept_txts,
            scores=kept_scores,
            word_results=(),
            elapse_list=[None, None, elapsed],
        )

    def _check_engine_choice(self) -> None:
        for section in ("Det", "Rec", "Cls"):
            choice = _enum_value(self.cfg[section].get("engine_type", "onnxruntime"))
            if choice not in _NATIVE_ENGINES:
                raise NotImplementedError(
                    f"{section}.engine_type={choice!r} is not available in the native "
                    "PP-OCRv6 build. Use EngineType.ONNXRUNTIME or EngineType.PPOCR_CPP."
                )
            version = _enum_value(self.cfg[section].get("ocr_version", "PP-OCRv6"))
            if section != "Cls" and version not in (OCRVersion.PPOCRV6.value, "PP-OCRv6", None):
                raise NotImplementedError(
                    f"The native engine runs PP-OCRv6 ONNX models, not {version}."
                )

    def _ensure_engine(self) -> _native.NativeOCR:
        if self._engine is not None:
            return self._engine
        det, rec, dictionary = _resolve_models(self.cfg)
        options = _native.default_options()
        options.backend = _native.PPOCR_BACKEND_CPU
        options.det_threshold = float(self.cfg["Det"].get("thresh", options.det_threshold))
        options.det_box_threshold = float(self.cfg["Det"].get("box_thresh", options.det_box_threshold))
        options.det_unclip_ratio = float(self.cfg["Det"].get("unclip_ratio", options.det_unclip_ratio))
        limit = self.cfg["Det"].get("limit_side_len")
        if limit:
            options.det_limit_side_len = int(limit)
        batch = self.cfg["Rec"].get("rec_batch_num")
        if batch:
            options.rec_batch_size = int(batch)
        self._engine = _native.NativeOCR(det, rec, dictionary, options)
        return self._engine


def _load_config(config_path: Optional[str], params: Optional[Dict[str, Any]]) -> ConfigNode:
    cfg = ParseParams.load(config_path)
    if params:
        ParseParams.update_batch(cfg, params)
    return cfg


def _prepare_image(engine: RapidOCR, original: np.ndarray):
    image = original
    ratio_w = ratio_h = 1.0
    pad_top = 0
    if bool(engine.cfg.Global.get("use_preprocess_img", True)):
        image, ratio_h, ratio_w = resize_image_within_bounds(
            image, float(engine.min_side_len), float(engine.max_side_len)
        )
    if bool(engine.cfg.Global.get("use_vertical_padding", True)):
        image, pad_top = apply_vertical_padding(
            image, float(engine.width_height_ratio), float(engine.min_height)
        )
    return image, ratio_w, ratio_h, pad_top


def _resolve_models(cfg: ConfigNode) -> tuple:
    det = cfg.Det.get("model_path")
    rec = cfg.Rec.get("model_path")
    dictionary = cfg.Rec.get("rec_keys_path")
    if det and rec and dictionary:
        return str(det), str(rec), str(dictionary)
    model_type = _enum_value(cfg.Rec.get("model_type") or cfg.Det.get("model_type") or "small")
    if model_type == "mobile":
        model_type = "small"
    root = cfg.Global.get("model_root_dir")
    fetched = download_models(str(model_type), str(root) if root else None)
    return det or fetched["det"], rec or fetched["rec"], dictionary or fetched["dict"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RapidOCR native PP-OCRv6 demo")
    parser.add_argument("image")
    parser.add_argument("--det", default=None)
    parser.add_argument("--rec", default=None)
    parser.add_argument("--dict", dest="dictionary", default=None)
    parser.add_argument("--model-type", default="tiny", choices=["tiny", "small", "medium"])
    args = parser.parse_args()
    params = {
        "Det.model_type": args.model_type,
        "Rec.model_type": args.model_type,
        "Global.use_cls": False,
    }
    if args.det:
        params["Det.model_path"] = args.det
    if args.rec:
        params["Rec.model_path"] = args.rec
    if args.dictionary:
        params["Rec.rec_keys_path"] = args.dictionary
    engine = RapidOCR(params=params)
    result = engine(args.image)
    print(result)
    if result.txts:
        for text, score in zip(result.txts, result.scores or ()):
            print(f"{text}\t{score:.4f}")


if __name__ == "__main__":
    main()
