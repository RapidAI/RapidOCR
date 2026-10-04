# C, C++, and Python API

The inference core is unchanged: a built-in PP-OCRv6 ONNX interpreter with
CPU and optional Vulkan execution. This document is the public surface other
programs should call.

## Build and install

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix "$PWD/install"
```

That produces:

- `libppocr.a` (`ppocr::static`)
- `libppocr.so` (`ppocr::ppocr`), SONAME `libppocr.so.5`
- headers under `include/ppocr/`
- `lib/cmake/ppocr/ppocrConfig.cmake`

```cmake
find_package(ppocr REQUIRED)
target_link_libraries(app PRIVATE ppocr::ppocr)   # shared
# or ppocr::static
```

`PPOCR_BUILD_SHARED` and `PPOCR_BUILD_STATIC` both default to `ON`.
Vulkan shaders are embedded only when a host `glslangValidator` (or the
vendored Windows `glslang.exe` on Windows) is available. The CPU engine does
not need them.

Examples:

```bash
./build/ppocr_c_api_demo det.onnx rec.onnx dict.txt image.ppm
./build/ppocr_cpp_api_demo det.onnx rec.onnx dict.txt image.ppm
./build/ppocr_c_api_demo --cpu-info
```

Images passed to the library are packed RGB, row-major, 8-bit. `ppocr_load_ppm`
reads binary P6 PPM. JPEG/PNG decoding stays in the application (or in the
Python package, which uses Pillow).

## C ABI

`include/ppocr/ppocr.h` is a stable C ABI:

- `extern "C"`
- opaque `ppocr_ocr*`
- plain structs for options, images, and text boxes
- integer status codes (`PPOCR_OK`, `PPOCR_ERR_*`)
- `ppocr_create` / `ppocr_destroy`, `ppocr_result_free`, `ppocr_image_free`
- thread-local `ppocr_last_error()`
- `ppocr_options.struct_size` so new fields can be appended

`ppocr_options_init` fills the same defaults as `ppocr::Options`. One handle
must not be used from two threads at once. `ppocr_recognize_batch` preserves
input order.

Runtime CPU selection is reported by `ppocr_query_cpu_info`. `active_isa` is
`avx512`, `avx2`, `neon`, or `scalar`. See `docs/PERFORMANCE.md` for the rule
that keeps AVX-512 instructions out of an AVX2-only process.

## C++ API

`include/ppocr/ppocr.hpp` is the C++ API already used by the in-tree tools:

```cpp
ppocr::Options options;
options.backend = ppocr::Backend::cpu_only;
ppocr::OCR ocr(det_path, rec_path, dict_path, options);
for (const ppocr::Result& result : ocr.Recognize(ppocr::LoadPPM(path))) {
  // result.text, result.confidence, result.bbox, result.box
}
```

`QueryCpuInfo()` and `SetDetectionThresholds()` are part of that header.
Exceptions use `std::runtime_error`. The C ABI catches them.

## Python package

This is **rapidocr 5.0.0a1**, the next major version of the PyPI package
`rapidocr`. The import name stays `rapidocr`. The 3.x line (through tag
`v3.9.2`) remains on `main`. There is no 4.x tag in this repository. A
migration note is in `python/README.md`.

The call shape follows 3.x:

```python
from rapidocr import RapidOCR, EngineType, ModelType, OCRVersion

engine = RapidOCR(params={
    "Det.model_path": "PP-OCRv6_det_tiny.onnx",
    "Rec.model_path": "PP-OCRv6_rec_tiny.onnx",
    "Rec.rec_keys_path": "ppocrv6_tiny_dict.txt",
    "Det.model_type": ModelType.TINY,
    "Rec.model_type": ModelType.TINY,
    "Det.ocr_version": OCRVersion.PPOCRV6,
    "Global.text_score": 0.5,
    "Global.use_cls": False,
})
result = engine("page.png", box_thresh=0.5, unclip_ratio=1.6)
result.txts
result.scores
result.boxes          # float32 array, shape (N, 4, 2), TL TR BR BL
result.elapse         # seconds
result.to_json()
result.to_markdown()
result.vis("vis.png")
```

`params` keys use the same `Section.name` form as `main` (`Det.box_thresh`,
`Global.text_score`, `Rec.rec_batch_num`, ...). A YAML `config_path` is
loaded with PyYAML and then overridden by `params`.

File paths, URLs, bytes, and PIL images are decoded as RGB. A numpy ndarray
is treated as BGR, which is what `main` does for `cv2` images, and converted
to RGB before the native engine.

### Binding choice

Python calls the C ABI with ctypes. One shared library is the contract for C,
for other FFI languages, and for Python, so the result struct cannot drift
from a second pybind11 wrapper. scikit-build-core still builds that library
into a wheel:

```bash
pip install ./python
```

### What does not match `main`

| `main` | this package |
| --- | --- |
| ONNX Runtime, OpenVINO, Paddle, PyTorch, TensorRT, MNN | built-in PP-OCRv6 interpreter only (`EngineType.ONNXRUNTIME` or `EngineType.PPOCR_CPP`) |
| PP-OCRv4 and v5 model zoo | PP-OCRv6 ONNX detector + recognizer |
| angle classifier (`use_cls=True` by default) | flag is accepted and ignored, with one warning |
| `use_det=False` / `use_rec=False` | raises `RapidOCRError` |
| word and character boxes | `return_word_box` warns; `word_results` is empty |
| `elapse_list` is `[det, cls, rec]` | `[None, None, total]` because the native call is one timed region |
| OpenCV drawing in `vis` | Pillow polygons |

`result.txts`, `result.scores`, and `result.boxes` stay the fields existing
callers read. `text_score`, `box_thresh`, and `unclip_ratio` still filter or
retune a call without rebuilding the models.
