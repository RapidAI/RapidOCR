# -*- encoding: utf-8 -*-
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import yaml

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


class _AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _to_attr_dict(obj):
    if isinstance(obj, dict):
        return _AttrDict({k: _to_attr_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr_dict(v) for v in obj]
    return obj


def _module_unavailable(name: str) -> bool:
    return name not in sys.modules and importlib.util.find_spec(name) is None


def _install_missing_runtime_stubs(monkeypatch):
    if _module_unavailable("cv2"):
        monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))

    if _module_unavailable("colorlog"):
        colorlog_mod = types.ModuleType("colorlog")

        class ColoredFormatter:
            def __init__(self, *args, **kwargs):
                pass

        colorlog_mod.ColoredFormatter = ColoredFormatter
        monkeypatch.setitem(sys.modules, "colorlog", colorlog_mod)

    if _module_unavailable("omegaconf"):
        omegaconf_mod = types.ModuleType("omegaconf")

        class OmegaConf:
            @staticmethod
            def load(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    return _to_attr_dict(yaml.safe_load(f))

            @staticmethod
            def select(cfg, key):
                cur = cfg
                for part in key.split("."):
                    if not isinstance(cur, dict) or part not in cur:
                        return None
                    cur = cur[part]
                return cur

        omegaconf_mod.DictConfig = _AttrDict
        omegaconf_mod.OmegaConf = OmegaConf
        monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_mod)

    if _module_unavailable("openvino"):
        openvino_mod = types.ModuleType("openvino")
        openvino_mod.Core = object
        monkeypatch.setitem(sys.modules, "openvino", openvino_mod)


class _FakeTensor:
    def __init__(self, data: np.ndarray):
        self.data = data


class _FakeInferRequest:
    def __init__(self, outputs):
        self._outputs = outputs
        self.last_inputs = None
        self.requested_indices = []

    def infer(self, inputs):
        self.last_inputs = inputs

    def get_output_tensor(self, index=None):
        self.requested_indices.append(index)
        if index is None:
            if len(self._outputs) != 1:
                raise RuntimeError(
                    "Check 'outputs.size() == 1' failed: "
                    "get_output_tensor() must be called on a function with exactly one parameter."
                )
            return _FakeTensor(self._outputs[0])
        return _FakeTensor(self._outputs[index])


def _import_openvino_main(monkeypatch):
    _install_missing_runtime_stubs(monkeypatch)
    from rapidocr.inference_engine.openvino import main as openvino_main

    return openvino_main


def test_openvino_single_output_returns_first_tensor(monkeypatch):
    openvino_main = _import_openvino_main(monkeypatch)

    expected = np.ones((1, 1, 8, 8), dtype=np.float32)
    fake_request = _FakeInferRequest([expected])

    session = object.__new__(openvino_main.OpenVINOInferSession)
    session.session = fake_request

    inp = np.zeros((1, 3, 8, 8), dtype=np.float32)
    actual = session(inp)

    assert np.array_equal(actual, expected)
    assert fake_request.last_inputs == [inp]
    assert fake_request.requested_indices == [0]


def test_openvino_multi_output_returns_first_tensor(monkeypatch):
    openvino_main = _import_openvino_main(monkeypatch)

    primary = np.full((1, 1, 8, 8), 2.0, dtype=np.float32)
    aux = np.zeros((1, 1, 8, 8), dtype=np.float32)
    fake_request = _FakeInferRequest([primary, aux])

    session = object.__new__(openvino_main.OpenVINOInferSession)
    session.session = fake_request

    inp = np.zeros((1, 3, 8, 8), dtype=np.float32)
    actual = session(inp)

    assert np.array_equal(actual, primary)
    assert fake_request.requested_indices == [0]
