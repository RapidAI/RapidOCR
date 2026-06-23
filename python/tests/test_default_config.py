# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

import pytest

from rapidocr import RapidOCR
from rapidocr.inference_engine.base import FileInfo, InferSession
from rapidocr.utils.parse_parameters import ParseParams
from rapidocr.utils.typings import (
    EngineType,
    LangCls,
    LangDet,
    LangRec,
    ModelType,
    OCRVersion,
    TaskType,
)

root_dir = Path(__file__).resolve().parent.parent
config_path = root_dir / "rapidocr" / "config.yaml"
tests_dir = root_dir / "tests" / "test_files"


@pytest.fixture()
def default_cfg():
    return ParseParams.load(config_path)


@pytest.fixture(scope="module")
def default_engine():
    return RapidOCR()


def get_default_model_info(model_cfg):
    return InferSession.get_model_url(
        FileInfo(
            model_cfg.engine_type,
            model_cfg.ocr_version,
            model_cfg.task_type,
            model_cfg.lang_type,
            model_cfg.model_type,
        )
    )


def test_default_global_config(default_cfg):
    assert default_cfg.Global.text_score == 0.5
    assert default_cfg.Global.use_det is True
    assert default_cfg.Global.use_cls is True
    assert default_cfg.Global.use_rec is True
    assert default_cfg.Global.min_height == 30
    assert default_cfg.Global.width_height_ratio == 8
    assert default_cfg.Global.max_side_len == 2000
    assert default_cfg.Global.min_side_len == 30
    assert default_cfg.Global.return_word_box is False
    assert default_cfg.Global.return_single_char_box is False
    assert default_cfg.Global.font_path is None
    assert default_cfg.Global.log_level == "info"
    assert default_cfg.Global.model_root_dir is None


def test_default_onnxruntime_engine_config(default_cfg):
    cfg = default_cfg.EngineConfig.onnxruntime

    assert cfg.intra_op_num_threads == -1
    assert cfg.inter_op_num_threads == -1
    assert cfg.enable_cpu_mem_arena is False
    assert cfg.cpu_ep_cfg.arena_extend_strategy == "kSameAsRequested"
    assert cfg.use_cuda is False
    assert cfg.cuda_ep_cfg.device_id == 0
    assert cfg.cuda_ep_cfg.arena_extend_strategy == "kNextPowerOfTwo"
    assert cfg.cuda_ep_cfg.cudnn_conv_algo_search == "EXHAUSTIVE"
    assert cfg.cuda_ep_cfg.do_copy_in_default_stream is True
    assert cfg.use_dml is False
    assert cfg.dml_ep_cfg is None
    assert cfg.use_cann is False
    assert cfg.cann_ep_cfg.device_id == 0
    assert cfg.cann_ep_cfg.npu_mem_limit == 21474836480
    assert cfg.cann_ep_cfg.op_select_impl_mode == "high_performance"
    assert cfg.cann_ep_cfg.optypelist_for_implmode == "Gelu"
    assert cfg.cann_ep_cfg.enable_cann_graph is True
    assert cfg.use_coreml is False
    assert cfg.coreml_ep_cfg.ModelFormat == "MLProgram"
    assert cfg.coreml_ep_cfg.MLComputeUnits == "ALL"
    assert cfg.coreml_ep_cfg.ModelCacheDirectory == "/tmp/RapidOCR"


def test_default_openvino_engine_config(default_cfg):
    cfg = default_cfg.EngineConfig.openvino

    assert cfg.inference_num_threads == -1
    assert cfg.performance_hint is None
    assert cfg.performance_num_requests == -1
    assert cfg.enable_cpu_pinning is None
    assert cfg.num_streams == -1
    assert cfg.enable_hyper_threading is None
    assert cfg.scheduling_core_type is None


def test_default_other_engine_configs(default_cfg):
    assert default_cfg.EngineConfig.paddle.cpu_math_library_num_threads == -1
    assert default_cfg.EngineConfig.paddle.use_npu is False
    assert default_cfg.EngineConfig.paddle.npu_ep_cfg.device_id == 0
    assert default_cfg.EngineConfig.paddle.use_cuda is False
    assert default_cfg.EngineConfig.paddle.cuda_ep_cfg.device_id == 0
    assert default_cfg.EngineConfig.paddle.cuda_ep_cfg.gpu_mem == 500

    assert default_cfg.EngineConfig.torch.use_cuda is False
    assert default_cfg.EngineConfig.torch.cuda_ep_cfg.device_id == 0
    assert default_cfg.EngineConfig.torch.use_npu is False
    assert default_cfg.EngineConfig.torch.npu_ep_cfg.device_id == 0
    assert default_cfg.EngineConfig.torch.use_mps is False

    assert default_cfg.EngineConfig.tensorrt.device_id == 0
    assert default_cfg.EngineConfig.tensorrt.use_fp16 is True
    assert default_cfg.EngineConfig.tensorrt.use_int8 is False
    assert default_cfg.EngineConfig.tensorrt.workspace_size == 1073741824
    assert default_cfg.EngineConfig.tensorrt.cache_dir is None
    assert default_cfg.EngineConfig.tensorrt.force_rebuild is False
    assert default_cfg.EngineConfig.tensorrt.det_profile.opt_shape == [
        1,
        3,
        736,
        736,
    ]
    assert default_cfg.EngineConfig.tensorrt.rec_profile.opt_shape == [
        6,
        3,
        48,
        320,
    ]
    assert default_cfg.EngineConfig.tensorrt.cls_profile.opt_shape == [
        6,
        3,
        48,
        192,
    ]

    assert default_cfg.EngineConfig.mnn == {}


@pytest.mark.parametrize(
    "section,task_type,expected_model_name,expected_sha256",
    [
        (
            "Det",
            TaskType.DET,
            "PP-OCRv6_det_small.onnx",
            "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
        ),
        (
            "Rec",
            TaskType.REC,
            "PP-OCRv6_rec_small.onnx",
            "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
        ),
    ],
)
def test_default_ppocrv6_model_config(
    default_cfg, section, task_type, expected_model_name, expected_sha256
):
    model_cfg = default_cfg[section]

    assert model_cfg.engine_type == EngineType.ONNXRUNTIME
    assert model_cfg.ocr_version == OCRVersion.PPOCRV6
    assert model_cfg.model_type == ModelType.SMALL
    assert model_cfg.task_type == task_type

    model_info = get_default_model_info(model_cfg)

    assert model_info["model_dir"].endswith(
        f"/PP-OCRv6/{task_type.value}/{expected_model_name}"
    )
    assert model_info["SHA256"] == expected_sha256


def test_default_det_config(default_cfg):
    assert default_cfg.Det.engine_type == EngineType.ONNXRUNTIME
    assert default_cfg.Det.lang_type == LangDet.CH
    assert default_cfg.Det.model_type == ModelType.SMALL
    assert default_cfg.Det.ocr_version == OCRVersion.PPOCRV6
    assert default_cfg.Det.task_type == TaskType.DET
    assert default_cfg.Det.model_path is None
    assert default_cfg.Det.model_dir is None
    assert default_cfg.Det.limit_side_len == 736
    assert default_cfg.Det.limit_type == "min"
    assert default_cfg.Det.std == [0.5, 0.5, 0.5]
    assert default_cfg.Det.mean == [0.5, 0.5, 0.5]
    assert default_cfg.Det.thresh == 0.3
    assert default_cfg.Det.box_thresh == 0.5
    assert default_cfg.Det.max_candidates == 1000
    assert default_cfg.Det.unclip_ratio == 1.6
    assert default_cfg.Det.use_dilation is True
    assert default_cfg.Det.score_mode == "fast"


def test_default_cls_config(default_cfg):
    assert default_cfg.Cls.engine_type == EngineType.ONNXRUNTIME
    assert default_cfg.Cls.lang_type == LangCls.CH
    assert default_cfg.Cls.model_type == ModelType.MOBILE
    assert default_cfg.Cls.ocr_version == OCRVersion.PPOCRV4
    assert default_cfg.Cls.task_type == TaskType.CLS
    assert default_cfg.Cls.model_path is None
    assert default_cfg.Cls.model_dir is None
    assert default_cfg.Cls.cls_image_shape == [3, 48, 192]
    assert default_cfg.Cls.cls_batch_num == 6
    assert default_cfg.Cls.cls_thresh == 0.9
    assert default_cfg.Cls.label_list == ["0", "180"]


def test_default_rec_config(default_cfg):
    assert default_cfg.Rec.engine_type == EngineType.ONNXRUNTIME
    assert default_cfg.Rec.lang_type == LangRec.CH
    assert default_cfg.Rec.model_type == ModelType.SMALL
    assert default_cfg.Rec.ocr_version == OCRVersion.PPOCRV6
    assert default_cfg.Rec.task_type == TaskType.REC
    assert default_cfg.Rec.model_path is None
    assert default_cfg.Rec.model_dir is None
    assert default_cfg.Rec.rec_keys_path is None
    assert default_cfg.Rec.rec_img_shape == [3, 48, 320]
    assert default_cfg.Rec.rec_batch_num == 6


def test_default_cls_model_config(default_cfg):
    model_info = get_default_model_info(default_cfg.Cls)

    assert model_info["model_dir"].endswith(
        "/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    )
    assert (
        model_info["SHA256"]
        == "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"
    )


def test_default_engine_runtime_config(default_engine):
    assert default_engine.use_det is True
    assert default_engine.use_cls is True
    assert default_engine.use_rec is True
    assert default_engine.cfg.Det.ocr_version == OCRVersion.PPOCRV6
    assert default_engine.cfg.Det.model_type == ModelType.SMALL
    assert default_engine.cfg.Rec.ocr_version == OCRVersion.PPOCRV6
    assert default_engine.cfg.Rec.model_type == ModelType.SMALL
    assert default_engine.cfg.Cls.ocr_version == OCRVersion.PPOCRV4
    assert default_engine.cfg.Cls.model_type == ModelType.MOBILE
    assert default_engine.text_det.limit_side_len == 736
    assert default_engine.text_det.limit_type == "min"
    assert default_engine.text_cls.cls_image_shape == [3, 48, 192]
    assert default_engine.text_rec.rec_image_shape == [3, 48, 320]
    assert default_engine.text_rec.rec_batch_num == 6


def test_default_engine_real_full_pipeline(default_engine):
    result = default_engine(tests_dir / "ch_en_num.jpg")

    assert result.txts is not None
    assert len(result) == 18
    assert result.txts[:3] == ("正品促销", "大桶装更划算", "强力去污符合国标")


def test_default_engine_real_rec_only(default_engine):
    result = default_engine(
        tests_dir / "text_rec.jpg",
        use_det=False,
        use_cls=False,
        use_rec=True,
    )

    assert result.txts is not None
    assert len(result) == 1
    assert result.txts[0] == "韩国小馆"
    assert result.scores[0] > 0.99
