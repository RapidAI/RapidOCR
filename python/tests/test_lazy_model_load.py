# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import threading
import time

import pytest

import rapidocr.main as main_module
from rapidocr import RapidOCR

THREAD_COUNT = 8


@pytest.mark.parametrize(
    ("loader_name", "model_attr", "ctor_name"),
    [
        ("_load_det_model", "text_det", "TextDetector"),
        ("_load_cls_model", "text_cls", "TextClassifier"),
        ("_load_rec_model", "text_rec", "TextRecognizer"),
    ],
)
def test_lazy_model_load_constructs_once_under_concurrent_first_use(
    monkeypatch, loader_name, model_attr, ctor_name
):
    engine = RapidOCR()
    assert getattr(engine, model_attr) is None

    constructed = []
    first_ctor_started = threading.Event()
    release_ctors = threading.Event()
    start_barrier = threading.Barrier(THREAD_COUNT)
    errors = []
    results = []

    class BlockingModel:
        def __init__(self, cfg):
            constructed.append(self)
            first_ctor_started.set()
            # Keep the instance unset until every racer has tried to construct.
            release_ctors.wait(timeout=5)

    monkeypatch.setattr(main_module, ctor_name, BlockingModel)

    def worker():
        try:
            start_barrier.wait(timeout=5)
            results.append(getattr(engine, loader_name)())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(THREAD_COUNT)]
    for thread in threads:
        thread.start()

    assert first_ctor_started.wait(timeout=5)
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline and len(constructed) < THREAD_COUNT:
        time.sleep(0.01)

    release_ctors.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert errors == []
    assert len(constructed) == 1
    assert len(results) == THREAD_COUNT
    assert all(result is constructed[0] for result in results)
    assert getattr(engine, model_attr) is constructed[0]
