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

If model paths are omitted, `RapidOCR()` downloads the same files as 3.x:
`default_models.yaml` selects the URL, the file is saved under
`Global.model_root_dir` (the package `models/` directory when that is null)
using the URL filename, and the SHA256 is checked. `download_models()` and
`download_models(config_yaml)` follow 3.x and return `None`. PP-OCRv6 ONNX
recognizers also download the dictionary published next to the MNN bundles
(`ppocrv6_tiny_dict.txt` / `ppocrv6_dict.txt`).

Build from a checkout (CMake and a C++20 compiler required). The repository
root is the installable project, so the sdist contains the native sources:

```bash
pip install .
```

`pip install ./python` also works from a checkout, because that project
points CMake at the parent tree. It does not produce a self-contained sdist.

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
| Angle classifier (`use_cls=True` by default) | The cls file is downloaded like 3.x. The native engine does not run it, and logs one warning. |
| `use_det=False` or `use_rec=False` | Raises `RapidOCRError`. Detection and recognition always run together. |
| Word and character boxes | `return_word_box` / `return_single_char_box` warn. `word_results` is empty. |
| `elapse_list` is `[det, cls, rec]` | `[None, None, total]`. The native call is one timed region. An empty page is `RapidOCROutput()` (`boxes`/`txts`/`img` are `None`), same as 3.x. |
| `vis()` draws with OpenCV in BGR | `vis()` draws with Pillow and returns RGB. |
| `EngineConfig` (CUDA, OpenVINO, TensorRT, MNN, …) | The keys load and can be read. They are not applied. |
| Detector `mean`/`std`, `limit_type`, `limit_side_len`, `use_dilation` | Passed through the C ABI and applied. `mean`/`std` are B, G, R, the same order as an OpenCV image. Recognizer normalization defaults to 0.5/0.5, matching 3.x. |
| `Global.min_height`, vertical padding, `max_side_len` | Applied before the native call, and boxes are mapped back. |

A call that only reads `txts`, `scores`, and `boxes` from a PP-OCRv6 model
can switch by installing this package and leaving `use_cls` at its default.
Set `use_cls=False` to skip the warning. Do not pass `use_det=False` or an
older OCR version.

## Wheels and publishing

`.github/workflows/build-wheels.yml` builds the manylinux2014 wheels and the
sdist. It runs on pushes and pull requests to `V5_NG`, on tags named `v5*`,
and from the Actions “Run workflow” button. A normal push does not upload
anything to PyPI. Publishing happens only when a `v5*` tag is pushed.

`5.0.0a1` is a pre-release. `pip install rapidocr` keeps installing the 3.x
release on `main`. The 5.x file is installed with an explicit version, for
example `pip install rapidocr==5.0.0a1`.

How to turn on PyPI Trusted Publishing, which tag to push, and the Actions
permission the GitHub Release step needs are in
[`docs/RELEASING.md`](../docs/RELEASING.md).
