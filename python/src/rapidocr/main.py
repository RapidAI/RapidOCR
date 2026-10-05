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
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import _native
from .inference_engine.base import FileInfo, InferSession
from .output import RapidOCROutput, filter_by_score
from .typings import EngineType, OCRVersion
from .utils.load_image import LoadImage, LoadImageError
from .utils.parse_parameters import ConfigNode, ParseParams
from .utils.process_img import apply_vertical_padding, map_boxes_to_original, resize_image_within_bounds

logger = logging.getLogger("rapidocr")
_PACKAGE_DIR = Path(__file__).resolve().parent

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
        _apply_vulkan_options(self.cfg, options)
        options.det_threshold = float(self.cfg["Det"].get("thresh", options.det_threshold))
        options.det_box_threshold = float(self.cfg["Det"].get("box_thresh", options.det_box_threshold))
        options.det_unclip_ratio = float(self.cfg["Det"].get("unclip_ratio", options.det_unclip_ratio))
        limit = self.cfg["Det"].get("limit_side_len")
        if limit:
            options.det_limit_side_len = int(limit)
        limit_type = str(self.cfg["Det"].get("limit_type", "max")).lower()
        options.det_limit_type = 1 if limit_type == "min" else 0
        _copy_norm(options.det_mean, options.det_std, self.cfg["Det"].get("mean"), self.cfg["Det"].get("std"))
        _copy_norm(
            options.rec_mean,
            options.rec_std,
            self.cfg["Rec"].get("mean"),
            self.cfg["Rec"].get("std"),
            default=(0.5, 0.5, 0.5),
        )
        options.det_use_dilation = 1 if bool(self.cfg["Det"].get("use_dilation", False)) else 0
        candidates = self.cfg["Det"].get("max_candidates")
        if candidates:
            options.det_max_candidates = int(candidates)
        shape = self.cfg["Rec"].get("rec_img_shape")
        if shape and len(shape) >= 2:
            options.rec_height = int(shape[1])
        batch = self.cfg["Rec"].get("rec_batch_num")
        if batch:
            options.rec_batch_size = int(batch)
        self._engine = _native.NativeOCR(det, rec, dictionary, options)
        return self._engine


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _apply_vulkan_options(cfg: ConfigNode, options: _native.Options) -> None:
    """Honor EngineConfig.ppocr_cpp the way 3.x honors use_cuda.

    The default stays on CPU. ``use_vulkan`` or ``backend: vulkan`` uses the
    GPU graph when a compute device is already available, and otherwise logs
    a warning and keeps CPU. ``backend: hybrid`` is the C++ best-effort path.
    ``gpu_only`` in C++ still fails closed; this wrapper does not.
    """

    engine = cfg.get("EngineConfig")
    section = engine.get("ppocr_cpp") if isinstance(engine, ConfigNode) else None
    if not isinstance(section, ConfigNode):
        return
    use_vulkan = _as_bool(section.get("use_vulkan", False))
    backend_name = str(section.get("backend", "cpu")).strip().lower()
    try:
        device_index = int(section.get("device_index", -1))
    except (TypeError, ValueError):
        device_index = -1
    options.vulkan_device_index = device_index
    wants_vulkan = use_vulkan or backend_name in {"vulkan", "gpu", "gpu_only"}
    wants_hybrid = backend_name == "hybrid"
    if not wants_vulkan and not wants_hybrid and device_index < 0:
        return
    if device_index >= 0:
        _native.request_vulkan_device(device_index)
    info = _native.backend_info()
    if wants_hybrid:
        options.backend = _native.PPOCR_BACKEND_HYBRID
        if not info["vulkan_compute_available"]:
            logger.warning("Vulkan hybrid was requested but no compute device is available; using CPU")
        return
    if wants_vulkan:
        if info["vulkan_compute_available"]:
            options.backend = _native.PPOCR_BACKEND_GPU
            logger.info("Vulkan device: %s", info["device_name"] or "(unnamed)")
        else:
            logger.warning(
                "Vulkan was requested but no compute device is available; using the CPU backend"
            )
            options.backend = _native.PPOCR_BACKEND_CPU


def _copy_norm(mean_out, std_out, mean, std, default=None) -> None:
    """Copy a length-3 mean/std into the C options. Missing values stay at the ABI default."""

    def take(slot, value):
        if value is None:
            if default is None:
                return
            value = default
        values = list(value)
        if len(values) != 3:
            return
        for index, item in enumerate(values):
            slot[index] = float(item)

    take(mean_out, mean)
    take(std_out, std)


def _load_config(config_path: Optional[str], params: Optional[Dict[str, Any]]) -> ConfigNode:
    cfg = ParseParams.load(config_path)
    if params:
        ParseParams.update_batch(cfg, params)
    if cfg.Global.get("model_root_dir") is None:
        cfg.Global.model_root_dir = _PACKAGE_DIR / "models"
    return cfg


def _section_info(section: ConfigNode) -> FileInfo:
    return FileInfo(
        section.engine_type,
        section.ocr_version,
        section.task_type,
        section.lang_type,
        section.model_type,
    )


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
    """Download missing det/rec/dict the way 3.x InferSession does.

    Files land in ``Global.model_root_dir`` under the URL filename. A cls
    model is downloaded when ``use_cls`` is set, matching 3.x, and is not
    passed to the native engine.
    """

    from .utils.download_file import DownloadFile, DownloadFileInput
    from .utils.download_models import download_task
    from .utils.log import logger as download_logger

    root = Path(cfg.Global.model_root_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    det = cfg.Det.get("model_path")
    rec = cfg.Rec.get("model_path")
    dictionary = cfg.Rec.get("rec_keys_path")

    if not det:
        info = _section_info(cfg.Det)
        download_task(root, info)
        det = root / Path(InferSession.get_model_url(info)["model_dir"]).name

    if not rec or not dictionary:
        info = _section_info(cfg.Rec)
        url_info = InferSession.get_model_url(info)
        if not rec:
            download_task(root, info)
            rec = root / Path(url_info["model_dir"]).name
        if not dictionary:
            dict_url = url_info.get("dict_url")
            if not dict_url:
                raise ValueError("recognizer model has no dict_url in default_models.yaml")
            dictionary = root / Path(dict_url).name
            if not dictionary.is_file():
                DownloadFile.run(
                    DownloadFileInput(
                        file_url=dict_url,
                        save_path=dictionary,
                        logger=download_logger,
                        sha256=None,
                    )
                )

    if bool(cfg.Global.get("use_cls", True)) and not cfg.Cls.get("model_path"):
        download_task(root, _section_info(cfg.Cls))

    return str(det), str(rec), str(dictionary)


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
