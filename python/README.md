# rapidocr 5

`rapidocr` 5.0.0a1 is the next major version of the PyPI package
[rapidocr](https://pypi.org/project/rapidocr/). The import name is still
`rapidocr`. This line is the native PP-OCRv6 engine on the `V5_NG` branch.
The 3.x line (latest tag `v3.9.2`) stays on `main`. This repository has no
4.x release.

```python
from rapidocr import RapidOCR, __version__

engine = RapidOCR(params={
    "Det.model_path": "PP-OCRv6_det_tiny.onnx",
    "Rec.model_path": "PP-OCRv6_rec_tiny.onnx",
    "Rec.rec_keys_path": "ppocrv6_tiny_dict.txt",
    "Global.use_cls": False,
})
result = engine("page.png")
print(__version__)
print(result.txts, result.scores)
print(result.to_json())
```

If model paths are omitted, the package downloads the PP-OCRv6 ONNX bundle
for `Det.model_type` / `Rec.model_type` (`tiny`, `small`, or `medium`).

Build a wheel from a checkout that already has the Vulkan-Headers submodule:

```bash
pip install scikit-build-core
pip install ./python
```

The binding is ctypes over the C ABI (`include/ppocr/ppocr.h`), so the wheel
ships `libppocr` and does not add a second C++ wrapper.

## Migrating from rapidocr 3.x

`RapidOCR(config_path=..., params={"Section.key": value})` and
`engine(img, use_det=..., use_cls=..., use_rec=..., text_score=..., box_thresh=..., unclip_ratio=...)`
keep the same names. `result.txts`, `result.scores`, `result.boxes`,
`result.elapse`, `to_json()`, `to_markdown()`, and `vis()` are still there.
File paths, URLs, bytes, and PIL images are RGB. A numpy array is treated as
BGR, then swapped, which is what 3.x does for OpenCV images.

What does not carry over from 3.x:

| 3.x (`main`) | rapidocr 5 |
| --- | --- |
| ONNX Runtime, OpenVINO, Paddle, PyTorch, TensorRT, MNN | Built-in PP-OCRv6 interpreter. `EngineType.ONNXRUNTIME` and `EngineType.PPOCR_CPP` both select it. The other engine names raise `NotImplementedError`. |
| PP-OCRv4 and PP-OCRv5 model zoo | PP-OCRv6 ONNX detector and recognizer only. |
| Angle classifier (`use_cls=True` by default) | The flag is accepted and ignored, with one warning. No cls model is downloaded. |
| `use_det=False` or `use_rec=False` | Raises `RapidOCRError`. Detection and recognition always run together. |
| Word and character boxes | `return_word_box` / `return_single_char_box` warn. `word_results` is empty. |
| `elapse_list` is `[det, cls, rec]` | `[None, None, total]`. The native call is one timed region. An empty page is `RapidOCROutput()` (`boxes`/`txts`/`img` are `None`), same as 3.x. |
| `vis()` draws with OpenCV in BGR | `vis()` draws with Pillow and returns RGB. |
| `EngineConfig` (CUDA, OpenVINO, TensorRT, MNN, …) | The keys load and can be read. They are not applied. |
| Detector `mean`/`std` of 0.5 | The native detector uses ImageNet mean/std. The same ONNX file can disagree with 3.x on boxes and scores. |
| `Global.min_height`, vertical padding, `max_side_len` | Applied before the native call, and boxes are mapped back. `limit_side_len` is still the detector's own limit. |

A call that only reads `txts`, `scores`, and `boxes` from a PP-OCRv6 model
can switch by installing this package and leaving `use_cls` at its default.
Set `use_cls=False` to skip the warning. Do not pass `use_det=False` or an
older OCR version.
