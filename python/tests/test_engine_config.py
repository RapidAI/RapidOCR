# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.inference_engine.onnxruntime import provider_config
from rapidocr.inference_engine.onnxruntime.provider_config import EP, ProviderConfig
from rapidocr.inference_engine.openvino.device_config import CPUConfig


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def make_provider_cfg(**overrides):
    cfg = AttrDict(
        use_cuda=False,
        cuda_ep_cfg={"device_id": 0},
        use_dml=False,
        dml_ep_cfg=None,
        use_cann=False,
        cann_ep_cfg={"device_id": 0},
        use_coreml=False,
        coreml_ep_cfg={"ModelFormat": "MLProgram"},
        cpu_ep_cfg={"arena_extend_strategy": "kSameAsRequested"},
    )
    cfg.update(overrides)
    return cfg


def make_provider_config(monkeypatch, cfg, providers, device="CPU"):
    monkeypatch.setattr(provider_config, "get_available_providers", lambda: providers)
    monkeypatch.setattr(provider_config, "get_device", lambda: device)
    return ProviderConfig(cfg)


def test_onnxruntime_provider_config_uses_cpu_by_default(monkeypatch):
    cfg = make_provider_cfg()
    provider_cfg = make_provider_config(
        monkeypatch,
        cfg,
        providers=[EP.CPU_EP.value],
    )

    assert provider_cfg.get_ep_list() == [(EP.CPU_EP.value, cfg.cpu_ep_cfg)]


def test_onnxruntime_provider_config_prefers_cuda_when_available(monkeypatch):
    cfg = make_provider_cfg(use_cuda=True)
    provider_cfg = make_provider_config(
        monkeypatch,
        cfg,
        providers=[EP.CUDA_EP.value, EP.CPU_EP.value],
        device="GPU",
    )

    assert provider_cfg.get_ep_list() == [
        (EP.CUDA_EP.value, cfg.cuda_ep_cfg),
        (EP.CPU_EP.value, cfg.cpu_ep_cfg),
    ]


def test_onnxruntime_provider_config_falls_back_when_cuda_unavailable(monkeypatch):
    cfg = make_provider_cfg(use_cuda=True)
    provider_cfg = make_provider_config(
        monkeypatch,
        cfg,
        providers=[EP.CPU_EP.value],
        device="CPU",
    )

    assert provider_cfg.get_ep_list() == [(EP.CPU_EP.value, cfg.cpu_ep_cfg)]


def test_onnxruntime_provider_config_prefers_directml_on_windows(monkeypatch):
    cfg = make_provider_cfg(use_dml=True, dml_ep_cfg={"device_id": 0})
    provider_cfg = make_provider_config(
        monkeypatch,
        cfg,
        providers=[EP.DIRECTML_EP.value, EP.CPU_EP.value],
    )
    monkeypatch.setattr(provider_config.platform, "system", lambda: "Windows")
    monkeypatch.setattr(provider_config.platform, "version", lambda: "10.0.19045")

    assert provider_cfg.get_ep_list() == [
        (EP.DIRECTML_EP.value, cfg.dml_ep_cfg),
        (EP.CPU_EP.value, cfg.cpu_ep_cfg),
    ]


def test_onnxruntime_provider_config_skips_directml_on_non_windows(monkeypatch):
    cfg = make_provider_cfg(use_dml=True)
    provider_cfg = make_provider_config(
        monkeypatch,
        cfg,
        providers=[EP.DIRECTML_EP.value, EP.CPU_EP.value],
    )
    monkeypatch.setattr(provider_config.platform, "system", lambda: "Linux")

    assert provider_cfg.get_ep_list() == [(EP.CPU_EP.value, cfg.cpu_ep_cfg)]


def test_onnxruntime_provider_config_adds_cann_when_available(monkeypatch):
    cfg = make_provider_cfg(use_cann=True)
    provider_cfg = make_provider_config(
        monkeypatch,
        cfg,
        providers=[EP.CANN_EP.value, EP.CPU_EP.value],
    )

    assert provider_cfg.get_ep_list() == [
        (EP.CANN_EP.value, cfg.cann_ep_cfg),
        (EP.CPU_EP.value, cfg.cpu_ep_cfg),
    ]


def test_onnxruntime_provider_config_adds_coreml_on_darwin(monkeypatch):
    cfg = make_provider_cfg(use_coreml=True)
    provider_cfg = make_provider_config(
        monkeypatch,
        cfg,
        providers=[EP.COREML_EP.value, EP.CPU_EP.value],
    )
    monkeypatch.setattr(provider_config.platform, "system", lambda: "Darwin")

    assert provider_cfg.get_ep_list() == [
        (EP.COREML_EP.value, cfg.coreml_ep_cfg),
        (EP.CPU_EP.value, cfg.cpu_ep_cfg),
    ]


def test_onnxruntime_provider_config_rejects_empty_session_providers(monkeypatch):
    cfg = make_provider_cfg()
    provider_cfg = make_provider_config(
        monkeypatch,
        cfg,
        providers=[EP.CPU_EP.value],
    )

    with pytest.raises(ValueError, match="Session Providers is empty"):
        provider_cfg.verify_providers([])


def test_openvino_cpu_config_omits_defaults():
    cfg = AttrDict(
        inference_num_threads=-1,
        performance_hint=None,
        performance_num_requests=-1,
        enable_cpu_pinning=None,
        num_streams=-1,
        enable_hyper_threading=None,
        scheduling_core_type=None,
    )

    assert CPUConfig(cfg).get_config() == {}


def test_openvino_cpu_config_maps_non_default_values(monkeypatch):
    cfg = AttrDict(
        inference_num_threads=4,
        performance_hint="LATENCY",
        performance_num_requests=2,
        enable_cpu_pinning=True,
        num_streams=1,
        enable_hyper_threading=False,
        scheduling_core_type="PCORE_ONLY",
    )
    monkeypatch.setattr("rapidocr.inference_engine.openvino.device_config.os.cpu_count", lambda: 8)

    assert CPUConfig(cfg).get_config() == {
        "INFERENCE_NUM_THREADS": "4",
        "PERFORMANCE_HINT": "LATENCY",
        "PERFORMANCE_HINT_NUM_REQUESTS": "2",
        "ENABLE_CPU_PINNING": "True",
        "NUM_STREAMS": "1",
        "ENABLE_HYPER_THREADING": "False",
        "SCHEDULING_CORE_TYPE": "PCORE_ONLY",
    }


def test_openvino_cpu_config_ignores_invalid_thread_count(monkeypatch):
    cfg = AttrDict(
        inference_num_threads=16,
        performance_hint=None,
        performance_num_requests=-1,
        enable_cpu_pinning=None,
        num_streams=-1,
        enable_hyper_threading=None,
        scheduling_core_type=None,
    )
    monkeypatch.setattr("rapidocr.inference_engine.openvino.device_config.os.cpu_count", lambda: 8)

    assert "INFERENCE_NUM_THREADS" not in CPUConfig(cfg).get_config()
