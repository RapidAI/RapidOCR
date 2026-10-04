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
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union
from urllib.request import urlopen

import numpy as np
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

from . import _native
from .download import download_models
from .output import RapidOCROutput, filter_by_score
from .typings import EngineType, ModelType, OCRVersion

logger = logging.getLogger("rapidocr")

_NATIVE_ENGINES = {EngineType.ONNXRUNTIME.value, EngineType.PPOCR_CPP.value, "onnxruntime", "ppocr_cpp"}
_SECTIONS = {"Global", "Det", "Rec", "Cls", "EngineConfig"}

_DEFAULTS: Dict[str, Any] = {
    "Global": {
        "text_score": 0.5,
        "use_det": True,
        "use_cls": True,
        "use_rec": True,
        "log_level": "info",
        "model_root_dir": None,
        "max_side_len": 2000,
        "min_side_len": 30,
    },
    "Det": {
        "engine_type": "onnxruntime",
        "lang_type": "ch",
        "model_type": "small",
        "ocr_version": "PP-OCRv6",
        "model_path": None,
        "thresh": 0.3,
        "box_thresh": 0.5,
        "unclip_ratio": 1.6,
        "limit_side_len": 736,
        "limit_type": "min",
    },
    "Rec": {
        "engine_type": "onnxruntime",
        "lang_type": "ch",
        "model_type": "small",
        "ocr_version": "PP-OCRv6",
        "model_path": None,
        "rec_keys_path": None,
        "rec_batch_num": 6,
    },
    "Cls": {
        "engine_type": "onnxruntime",
        "model_path": None,
        "cls_thresh": 0.9,
    },
    "EngineConfig": {},
}


class LoadImageError(Exception):
    pass


class RapidOCRError(Exception):
    pass


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


class RapidOCR:
    def __init__(self, config_path: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
        self.cfg = _load_config(config_path, params)
        level = str(self.cfg["Global"].get("log_level", "info")).upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        self.text_score = float(self.cfg["Global"]["text_score"])
        self.use_det = bool(self.cfg["Global"]["use_det"])
        self.use_cls = bool(self.cfg["Global"]["use_cls"])
        self.use_rec = bool(self.cfg["Global"]["use_rec"])
        self.return_word_box = False
        self._engine: Optional[_native.NativeOCR] = None
        self._warned_cls = False
        self._check_engine_choice()

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
        if use_det is not None:
            self.use_det = use_det
        if use_cls is not None:
            self.use_cls = use_cls
        if use_rec is not None:
            self.use_rec = use_rec
        if text_score is not None:
            self.text_score = float(text_score)
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
        rgb = _load_image(img_content)
        engine = self._ensure_engine()
        det_threshold = float(self.cfg["Det"].get("thresh", 0.3))
        box_threshold = float(box_thresh if box_thresh is not None else self.cfg["Det"].get("box_thresh", 0.5))
        unclip = float(unclip_ratio if unclip_ratio is not None else self.cfg["Det"].get("unclip_ratio", 1.6))
        engine.set_det_thresholds(det_threshold, box_threshold, unclip)
        boxes, txts, scores = engine.recognize(rgb)
        kept_boxes, kept_txts, kept_scores = filter_by_score(boxes, txts, scores, self.text_score)
        elapsed = time.perf_counter() - started
        if len(kept_txts) == 0:
            return RapidOCROutput(img=rgb, elapse_list=[None, None, elapsed])
        return RapidOCROutput(
            img=rgb,
            boxes=kept_boxes,
            txts=kept_txts,
            scores=kept_scores,
            word_results=tuple(() for _ in kept_txts),
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


def _load_config(config_path: Optional[str], params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    cfg = {key: dict(value) if isinstance(value, dict) else value for key, value in _DEFAULTS.items()}
    if config_path:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(config_path)
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("config root must be a mapping")
        for section, values in loaded.items():
            if section not in cfg or not isinstance(values, dict):
                cfg[section] = values
                continue
            cfg[section].update(values)
    if params:
        _apply_params(cfg, params)
    return cfg


def _apply_params(cfg: Dict[str, Any], params: Mapping[str, Any]) -> None:
    for key, value in params.items():
        parts = str(key).split(".")
        if len(parts) < 2 or not all(parts):
            raise ValueError(f"{key} is not a valid key.")
        if parts[0] not in _SECTIONS and parts[0] not in cfg:
            raise ValueError(f"{key} is not a valid key.")
        node = cfg.setdefault(parts[0], {})
        if not isinstance(node, dict):
            raise ValueError(f"{key} is not a valid key.")
        for part in parts[1:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"{key} is not a valid key.")
            node = child
        node[parts[-1]] = _enum_value(value)


def _resolve_models(cfg: Dict[str, Any]) -> tuple:
    det = cfg["Det"].get("model_path")
    rec = cfg["Rec"].get("model_path")
    dictionary = cfg["Rec"].get("rec_keys_path")
    if det and rec and dictionary:
        return str(det), str(rec), str(dictionary)
    model_type = _enum_value(cfg["Rec"].get("model_type") or cfg["Det"].get("model_type") or "small")
    if model_type == ModelType.MOBILE.value:
        model_type = "small"
    root = cfg["Global"].get("model_root_dir")
    fetched = download_models(str(model_type), str(root) if root else None)
    return det or fetched["det"], rec or fetched["rec"], dictionary or fetched["dict"]


def _load_image(img: Any) -> np.ndarray:
    origin = type(img)
    array = _to_array(img)
    return _to_rgb(array, origin)


def _to_array(img: Any) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        return np.asarray(ImageOps.exif_transpose(img) or img)
    if isinstance(img, (str, Path)):
        text = str(img)
        if text.startswith("http://") or text.startswith("https://"):
            with urlopen(text, timeout=60) as response:
                data = response.read()
            return _decode_bytes(data)
        path = Path(text)
        if not path.is_file():
            raise LoadImageError(f"{path} does not exist.")
        return _decode_bytes(path.read_bytes())
    if isinstance(img, (bytes, bytearray)):
        return _decode_bytes(bytes(img))
    raise LoadImageError(f"The img type {type(img)} is not supported")


def _decode_bytes(data: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image) or image
        return np.asarray(image)
    except UnidentifiedImageError as exc:
        raise LoadImageError("cannot identify image file") from exc


def _to_rgb(img: np.ndarray, origin: type) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")
    channels = img.shape[2]
    from_array = issubclass(origin, np.ndarray)
    if channels == 1:
        plane = img[:, :, 0]
        return np.stack([plane, plane, plane], axis=-1)
    if channels == 4:
        rgb = img[:, :, :3]
        if from_array:
            rgb = rgb[:, :, ::-1]
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        composited = rgb.astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return np.clip(composited, 0, 255).astype(np.uint8)
    if channels != 3:
        raise LoadImageError(f"The channel({channels}) of the img is not in [1, 2, 3, 4]")
    if from_array:
        # OpenCV-style ndarrays are BGR on main. The native engine wants RGB.
        return np.ascontiguousarray(img[:, :, ::-1])
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


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
