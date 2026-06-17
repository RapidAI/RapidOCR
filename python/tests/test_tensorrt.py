# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
import types
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.utils.typings import EngineType, LangDet, ModelType, OCRVersion, TaskType


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def install_fake_tensorrt_modules(monkeypatch):
    trt_module = types.ModuleType("tensorrt")

    class Logger:
        WARNING = 2

        def __init__(self, severity=None):
            self.severity = severity

    class TensorIOMode:
        INPUT = "input"
        OUTPUT = "output"

    trt_module.Logger = Logger
    trt_module.ICudaEngine = object
    trt_module.IExecutionContext = object
    trt_module.INetworkDefinition = object
    trt_module.IOptimizationProfile = object
    trt_module.TensorIOMode = TensorIOMode
    trt_module.nptype = lambda dtype: dtype
    trt_module.volume = lambda shape: 1

    cuda_module = types.ModuleType("cuda")
    bindings_module = types.ModuleType("cuda.bindings")
    runtime_module = types.ModuleType("cuda.bindings.runtime")

    class CudaStatus:
        value = 0

    class CudaDeviceAttr:
        cudaDevAttrComputeCapabilityMajor = "major"
        cudaDevAttrComputeCapabilityMinor = "minor"

    class CudaMemcpyKind:
        cudaMemcpyHostToDevice = "host_to_device"
        cudaMemcpyDeviceToHost = "device_to_host"

    runtime_module.cudaDeviceAttr = CudaDeviceAttr
    runtime_module.cudaMemcpyKind = CudaMemcpyKind
    runtime_module.cudaSetDevice = lambda device_id: (CudaStatus(),)
    runtime_module.cudaDeviceGetAttribute = lambda attr, device_id: (
        CudaStatus(),
        8 if attr == CudaDeviceAttr.cudaDevAttrComputeCapabilityMajor else 7,
    )
    runtime_module.cudaStreamSynchronize = lambda stream: (CudaStatus(),)
    runtime_module.cudaStreamDestroy = lambda stream: (CudaStatus(),)

    monkeypatch.setitem(sys.modules, "tensorrt", trt_module)
    monkeypatch.setitem(sys.modules, "cuda", cuda_module)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings_module)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", runtime_module)


class FakeEngine:
    def create_execution_context(self):
        return object()


def make_tensorrt_cfg(tmp_path, **overrides):
    model_root_dir = tmp_path / "models"
    model_root_dir.mkdir(exist_ok=True)

    cfg = AttrDict(
        engine_type=EngineType.TENSORRT,
        engine_cfg={},
        model_root_dir=model_root_dir,
        model_path=None,
        ocr_version=OCRVersion.PPOCRV4,
        task_type=TaskType.DET,
        lang_type=LangDet.EN,
        model_type=ModelType.MOBILE,
    )
    cfg.update(overrides)
    return cfg


def import_tensorrt_main(monkeypatch):
    install_fake_tensorrt_modules(monkeypatch)

    from rapidocr.inference_engine.tensorrt import main as tensorrt_main

    monkeypatch.setattr(
        tensorrt_main,
        "allocate_buffers",
        lambda engine, context: ([], [], [], None),
    )
    return tensorrt_main


def test_tensorrt_cache_dir_initializes_model_root_dir_for_build(
    monkeypatch, tmp_path
):
    tensorrt_main = import_tensorrt_main(monkeypatch)

    downloaded_paths = []

    def fake_download(input_params):
        save_path = Path(input_params.save_path)
        save_path.write_bytes(b"fake onnx model")
        downloaded_paths.append(save_path)

    class FakeEngineBuilder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build(self):
            return FakeEngine()

    monkeypatch.setattr(tensorrt_main.DownloadFile, "run", fake_download)
    monkeypatch.setattr(tensorrt_main, "TRTEngineBuilder", FakeEngineBuilder)

    cfg = make_tensorrt_cfg(tmp_path, engine_cfg={"cache_dir": tmp_path / "trt_cache"})

    session = tensorrt_main.TRTInferSession(cfg)

    assert session.model_root_dir == cfg.model_root_dir
    assert downloaded_paths == [cfg.model_root_dir / "en_PP-OCRv3_det_mobile.onnx"]


def test_tensorrt_default_cache_dir_uses_model_root_dir(monkeypatch, tmp_path):
    tensorrt_main = import_tensorrt_main(monkeypatch)

    engine_paths = []

    class FakeEngineBuilder:
        def __init__(self, **kwargs):
            engine_paths.append(kwargs["engine_path"])

        def build(self):
            return FakeEngine()

    monkeypatch.setattr(
        tensorrt_main.DownloadFile,
        "run",
        lambda input_params: Path(input_params.save_path).write_bytes(b"fake onnx"),
    )
    monkeypatch.setattr(tensorrt_main, "TRTEngineBuilder", FakeEngineBuilder)

    cfg = make_tensorrt_cfg(tmp_path)

    session = tensorrt_main.TRTInferSession(cfg)

    assert session.model_root_dir == cfg.model_root_dir
    assert engine_paths == [
        cfg.model_root_dir / "models" / "en_PP-OCRv4_det_mobile_sm87_fp16.engine"
    ]


def test_tensorrt_explicit_model_path_skips_download(monkeypatch, tmp_path):
    tensorrt_main = import_tensorrt_main(monkeypatch)

    model_path = tmp_path / "custom.onnx"
    model_path.write_bytes(b"fake onnx")
    builder_kwargs = []

    class FakeEngineBuilder:
        def __init__(self, **kwargs):
            builder_kwargs.append(kwargs)

        def build(self):
            return FakeEngine()

    def fail_download(input_params):
        raise AssertionError("explicit model_path should not download a default model")

    monkeypatch.setattr(tensorrt_main.DownloadFile, "run", fail_download)
    monkeypatch.setattr(tensorrt_main, "TRTEngineBuilder", FakeEngineBuilder)

    cfg = make_tensorrt_cfg(
        tmp_path,
        engine_cfg={"cache_dir": tmp_path / "trt_cache", "use_fp16": False},
        model_path=model_path,
    )

    tensorrt_main.TRTInferSession(cfg)

    assert builder_kwargs[0]["onnx_path"] == model_path
    assert builder_kwargs[0]["engine_path"] == (
        tmp_path / "trt_cache" / "custom_sm87_fp32.engine"
    )


def test_tensorrt_loads_cached_engine_without_rebuild(monkeypatch, tmp_path):
    tensorrt_main = import_tensorrt_main(monkeypatch)

    cfg = make_tensorrt_cfg(tmp_path, engine_cfg={"cache_dir": tmp_path / "trt_cache"})
    engine_path = tmp_path / "trt_cache" / "en_PP-OCRv4_det_mobile_sm87_fp16.engine"
    engine_path.parent.mkdir()
    engine_path.write_bytes(b"cached engine")

    loaded_paths = []

    def fake_load_engine(self, path):
        loaded_paths.append(path)
        return FakeEngine()

    def fail_build(*args, **kwargs):
        raise AssertionError("cached engine should be loaded without rebuilding")

    monkeypatch.setattr(tensorrt_main.TRTInferSession, "_load_engine", fake_load_engine)
    monkeypatch.setattr(tensorrt_main, "TRTEngineBuilder", fail_build)

    tensorrt_main.TRTInferSession(cfg)

    assert loaded_paths == [engine_path]


def test_tensorrt_force_rebuild_ignores_cached_engine(monkeypatch, tmp_path):
    tensorrt_main = import_tensorrt_main(monkeypatch)

    cfg = make_tensorrt_cfg(
        tmp_path,
        engine_cfg={"cache_dir": tmp_path / "trt_cache", "force_rebuild": True},
    )
    engine_path = tmp_path / "trt_cache" / "en_PP-OCRv4_det_mobile_sm87_fp16.engine"
    engine_path.parent.mkdir()
    engine_path.write_bytes(b"cached engine")

    build_count = 0

    class FakeEngineBuilder:
        def __init__(self, **kwargs):
            pass

        def build(self):
            nonlocal build_count
            build_count += 1
            return FakeEngine()

    def fail_load_engine(self, path):
        raise AssertionError("force_rebuild should skip cached engine loading")

    monkeypatch.setattr(
        tensorrt_main.DownloadFile,
        "run",
        lambda input_params: Path(input_params.save_path).write_bytes(b"fake onnx"),
    )
    monkeypatch.setattr(tensorrt_main.TRTInferSession, "_load_engine", fail_load_engine)
    monkeypatch.setattr(tensorrt_main, "TRTEngineBuilder", FakeEngineBuilder)

    tensorrt_main.TRTInferSession(cfg)

    assert build_count == 1


def test_tensorrt_falls_back_to_rebuild_when_cached_engine_load_fails(
    monkeypatch, tmp_path
):
    tensorrt_main = import_tensorrt_main(monkeypatch)

    cfg = make_tensorrt_cfg(tmp_path, engine_cfg={"cache_dir": tmp_path / "trt_cache"})
    engine_path = tmp_path / "trt_cache" / "en_PP-OCRv4_det_mobile_sm87_fp16.engine"
    engine_path.parent.mkdir()
    engine_path.write_bytes(b"bad cached engine")

    build_count = 0

    class FakeEngineBuilder:
        def __init__(self, **kwargs):
            pass

        def build(self):
            nonlocal build_count
            build_count += 1
            return FakeEngine()

    def fake_load_engine(self, path):
        raise RuntimeError("cannot deserialize")

    monkeypatch.setattr(
        tensorrt_main.DownloadFile,
        "run",
        lambda input_params: Path(input_params.save_path).write_bytes(b"fake onnx"),
    )
    monkeypatch.setattr(tensorrt_main.TRTInferSession, "_load_engine", fake_load_engine)
    monkeypatch.setattr(tensorrt_main, "TRTEngineBuilder", FakeEngineBuilder)

    tensorrt_main.TRTInferSession(cfg)

    assert build_count == 1


def test_tensorrt_square_padding_and_crop(monkeypatch):
    tensorrt_main = import_tensorrt_main(monkeypatch)

    session = object.__new__(tensorrt_main.TRTInferSession)
    session._closed = True
    session._max_square_size = 64

    import numpy as np

    img = np.ones((1, 3, 32, 48), dtype=np.float32)
    padded, original_hw = session._pad_to_square(img)

    assert padded.shape == (1, 3, 64, 64)
    assert original_hw == (32, 48)
    np.testing.assert_array_equal(padded[:, :, :32, :48], img)
    assert padded[:, :, 32:, :].sum() == 0
    assert padded[:, :, :, 48:].sum() == 0

    cropped = session._crop_output(padded, original_hw)
    assert cropped.shape == (1, 3, 32, 48)


def test_tensorrt_get_dict_key_url_uses_fallback_engines(monkeypatch):
    tensorrt_main = import_tensorrt_main(monkeypatch)

    from rapidocr.inference_engine.base import FileInfo
    from rapidocr.utils.typings import LangRec

    file_info = FileInfo(
        engine_type=EngineType.TENSORRT,
        ocr_version=OCRVersion.PPOCRV5,
        task_type=TaskType.REC,
        lang_type=LangRec.EN,
        model_type=ModelType.MOBILE,
    )

    dict_url = tensorrt_main.TRTInferSession.get_dict_key_url(file_info)

    assert dict_url.endswith("/ppocrv5_en_dict.txt")
