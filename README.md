<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Logov2_black.png"  width="60%" height="60%">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Logov2_white.png"  width="60%" height="60%">
    <img alt="RapidOCR" src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Logov2_white.png">
  </picture>

<div>&nbsp;</div>
<div align="center">
    <b><font size="4"><i>Open source OCR for the security of the digital world</i></font></b>
</div>
<div>&nbsp;</div>

<a href="https://clawhub.ai/rapidai/rapidocr" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A6%9E%20ClawHub-RapidOCR%20Skill-blue?link=https%3A%2F%2Fclawhub.ai%2Frapidai%2Frapidocr"></a>
<a href="https://huggingface.co/spaces/RapidAI/RapidOCRv3" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face Demo-blue"></a>
<a href="https://www.modelscope.cn/studios/RapidAI/RapidOCRv3.0.0/summary" target="_blank"><img src="https://img.shields.io/badge/魔搭-Demo-blue"></a>
<a href="https://colab.research.google.com/github/RapidAI/RapidOCR/blob/main/assets/RapidOCRDemo.ipynb" target="_blank"><img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/colab-badge.svg" alt="Open in Colab"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.9-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
<a href="https://github.com/RapidAI/RapidOCR/graphs/contributors"><img src="https://img.shields.io/github/contributors/RapidAI/RapidOCR?color=9ea"></a>
<a href="https://pepy.tech/project/rapidocr"><img src="https://static.pepy.tech/personalized-badge/rapidocr?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads%20rapidocr"></a>
<a href="https://pypi.org/project/rapidocr/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidocr"></a>
<a href="https://github.com/RapidAI/RapidOCR/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidOCR?color=ccf"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

Join our [Discord](https://discord.gg/33eyQJq498)

[简体中文](./README-CN.md) | English
</div>

This branch (`V5_NG`) is **rapidocr 5.0.0a1**: a C++ PP-OCRv6 engine with a C ABI, a C++ API, and a Python package whose import name is still `rapidocr`. The 3.x line (latest tag `v3.9.2`) stays on [`main`](https://github.com/RapidAI/RapidOCR/tree/main). Hosted demos linked from the badges above run that 3.x line.

### 📝 Introduction

RapidOCR is an open-source OCR library aimed at small models, offline deployment, and a short path from a PP-OCR weight file to a text box. Version 5 keeps the `RapidOCR()` call that 3.x applications already use, and runs PP-OCRv6 inside this repository instead of through ONNX Runtime.

The interpreter implements the operators used by the official PP-OCRv6 tiny, small, and medium ONNX detector and recognizer graphs: convolutions, GEMM, pooling, normalization, resize, shape ops, attention matmul, and the DB / CTC post-process. It is a PP-OCRv6 runtime, which means a general ONNX model still needs another engine.

If this project helps your work, a ⭐ Star helps it reach the next person.

### ✨ Features

- C++ PP-OCRv6 detector and recognizer. The library does not link ONNX Runtime, Paddle Inference, or OpenCV.
- One binary dispatches at runtime to AVX-512, AVX2+FMA, or AArch64 NEON. `PPOCR_FORCE_ISA=scalar|avx2|avx512|neon` selects a legal path for measurements. An AVX2-only machine never executes the AVX-512 object file.
- Optional Vulkan compute. Source builds embed SPIR-V when `third_party/Vulkan-Headers` and `glslangValidator` are present (`PPOCR_ENABLE_VULKAN` defaults to `ON`). The loader is `dlopen` / `LoadLibrary` of `libvulkan.so.1` or `vulkan-1.dll`, so Vulkan is not a `DT_NEEDED` dependency. Published wheels pass `-DPPOCR_ENABLE_VULKAN=OFF`.
- C ABI (`include/ppocr/ppocr.h`), C++ API (`include/ppocr/ppocr.hpp`), and the Python package `rapidocr`. Python calls the C ABI with ctypes.
- The Python call shape follows 3.x: `RapidOCR(params={"Section.key": value})`, then `result.txts`, `result.scores`, `result.boxes`, `result.elapse`, `to_json()`, `to_markdown()`, and `vis()`.
- Missing model files are downloaded from the ModelScope URLs in `default_models.yaml` (the same `v3.9.2` file names as 3.x) into `Global.model_root_dir`, and the SHA256 is checked.

### 🎥 Visualization

<div align="center">
    <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/demo.gif" alt="Demo" width="100%" height="100%">
</div>

The animation is the RapidOCR 3.x web demo. On this branch the checked line sample is `Hello RapidOCR 123` from [`python/tests/data/hello.png`](python/tests/data/hello.png) (960×240).

### ⚡ Performance

Models are PP-OCRv6 **tiny** ONNX. `use_cls=False`, so the angle classifier does not run.

| Sample | Size | What it contains |
| --- | --- | --- |
| hello | 960×240 | one line, `Hello RapidOCR 123` (`python/tests/data/hello.png`) |
| page | 1700×2200 | 45 lines. The page file is a local measurement image and is not committed in this repository. |

**x86, Python wall clock.** 4 vCPU Intel Xeon (family 6, model 207) with AVX2 and AVX-512F/DQ/BW/VL, constant 2.4 GHz, `g++` 13.3, Python 3.12. The default pool is 4 threads (`min(16, hardware_concurrency)`). Forced AVX2 is `PPOCR_FORCE_ISA=avx2` in the 5.0.0a1 process; ONNX Runtime 1.30.0 on this CPU stays on AVX-512. Each figure is the median of 8 calls after one warmup. Both packages use the shared Python defaults (`config.yaml`: short-side limit 736, mean/std 0.5, dilation on). A 960×240 line is enlarged until the short side reaches 736. `rapidocr==3.9.2`'s own published default is PP-OCRv6 **small** with the classifier left on; that configuration was not timed.

| Package | hello | page |
| --- | --- | --- |
| rapidocr 5.0.0a1, AVX-512 | 105.1 ms | 356.7 ms |
| rapidocr 5.0.0a1, AVX2 | 119.1 ms | 454.7 ms |
| rapidocr 3.9.2 (ORT 1.30.0) | 193.0 ms | 508.3 ms |

Against the 3.9.2 median, AVX-512 hello is 46% faster and the page is 30% faster. Forced AVX2 hello is 38% faster and the page is 11% faster. 3.9.2 was not remeasured for this table; 193.0 ms and 508.3 ms are the earlier back-to-back run. The page text is the same 45 strings on all three rows. Hello is `Hello RapidOCR 123`. Boxes versus the previous 5.0.0a1 binary have IoU 1.0 on both images.

**ARM, C++.** 20 cores: 10× Cortex-X925 (up to 3.9 GHz) and 10× Cortex-A725 (up to 2.8 GHz), `g++` 13.3, NEON. `ppocr_determinism` with `Backend::cpu_only`, 16 threads, and the C++ defaults (long-side limit 960, ImageNet detector mean/std, dilation off). AArch64 recognition widths stay exact.

| ISA | Threads | hello | page |
| --- | --- | --- | --- |
| NEON | 16 | 32.7 ms | 222.9 ms |

Determinism passed on both images (1 line and 45 lines).

The older x86 C++ long-side-960 rows and the 1-thread rows are not listed. Repeating `ppocr_determinism` on this VM after the current kernels moved the AVX-512 page mean across about 203–229 ms from one pass to the next, so those means are not published. `RapidOCR()` in Python keeps the short-side preprocess in the table above. Set `Det.limit_type`, `Det.limit_side_len`, `Det.mean`, and `Det.std` when you want the C++ defaults. Earlier kernel notes live in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) and are older than this section.

Vulkan page times from earlier adapter checks are not repeated here. Lavapipe on this VM is a software rasterizer, so its milliseconds are not GPU performance. `Backend::gpu_only` does not fall back to a CPU neural graph.

### 🛠️ Installation

`5.0.0a1` is a pre-release. On 2026-10-05 PyPI's newest `rapidocr` file is still 3.9.2, so both of the following commands install **3.9.2** until a `v5*` tag is published (see [Releasing](#-releasing)):

```bash
pip install rapidocr
pip install --pre rapidocr
```

After that tag, `pip install rapidocr` keeps resolving to the newest final release, and the pre-release is installed with:

```bash
pip install --pre rapidocr
```

From a checkout of this branch (CMake ≥ 3.20, a C++20 compiler, and the Python dependencies in `pyproject.toml`):

```bash
pip install .
```

That build follows `pyproject.toml`: shared `libppocr`, examples and tests off, Vulkan off. A local wheel is the same build packaged first:

```bash
pip wheel . -w dist
pip install dist/rapidocr-*.whl
```

`pip install ./python` also works from a checkout. It points CMake at the parent tree and does not produce a self-contained sdist. The repository-root `pip install .` does.

#### CMake

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix "$PWD/install"
```

`PPOCR_BUILD_SHARED` and `PPOCR_BUILD_STATIC` default to `ON`. Turn Vulkan off for a CPU-only library:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPPOCR_ENABLE_VULKAN=OFF
```

Turn it on for a source package (headers and `glslangValidator` required):

```bash
CMAKE_ARGS="-DPPOCR_ENABLE_VULKAN=ON -DPPOCR_BUILD_STATIC=OFF" pip install .
```

#### CMake package

```cmake
cmake_minimum_required(VERSION 3.20)
project(ocr_app C)
find_package(ppocr REQUIRED)
add_executable(ocr_app main.c)
target_link_libraries(ocr_app PRIVATE ppocr::ppocr)
```

The install prefix contains `libppocr.so` (SONAME `libppocr.so.5`), headers under `include/ppocr/`, and `lib/cmake/ppocr/ppocrConfig.cmake`. Link `ppocr::static` for the static library. At run time the dynamic loader has to see `libppocr.so.5` (`LD_LIBRARY_PATH` or an rpath).

### 📋 Usage

#### Python

Paths below are the PP-OCRv6 tiny files. With the paths omitted, `RapidOCR()` downloads the configured default (PP-OCRv6 small, unless `Det.model_type` / `Rec.model_type` is `tiny`).

```python
from rapidocr import RapidOCR

engine = RapidOCR(params={
    "Det.model_path": "PP-OCRv6_det_tiny.onnx",
    "Rec.model_path": "PP-OCRv6_rec_tiny.onnx",
    "Rec.rec_keys_path": "ppocrv6_tiny_dict.txt",
    "Global.use_cls": False,
})
result = engine("hello.png")
print(result.txts)
print(result.scores)
print(result.to_json())
result.vis("vis_result.jpg")
```

`result.boxes` is a float32 array of shape `(N, 4, 2)` in TL, TR, BR, BL order. File paths, URLs, bytes, and PIL images are read as RGB. A numpy ndarray is treated as BGR, matching 3.x OpenCV images, and converted to RGB before the native engine.

Automatic download of the tiny pair (files land in the package `models/` directory when `Global.model_root_dir` is null):

```python
from rapidocr import RapidOCR

engine = RapidOCR(params={
    "Det.model_type": "tiny",
    "Rec.model_type": "tiny",
    "Global.use_cls": False,
})
result = engine("hello.png")
print(result.txts)
```

#### Command line

```bash
python -m rapidocr hello.png \
  --det PP-OCRv6_det_tiny.onnx \
  --rec PP-OCRv6_rec_tiny.onnx \
  --dict ppocrv6_tiny_dict.txt \
  --model-type tiny
```

`--model-type` is `tiny`, `small`, or `medium`. The console script `rapidocr` calls the same `main`.

#### C++

```cpp
#include "ppocr/ppocr.hpp"

#include <iostream>

int main(int argc, char** argv) {
  ppocr::Options options;
  options.backend = ppocr::Backend::cpu_only;
  ppocr::OCR ocr(argv[1], argv[2], argv[3], options);
  for (const ppocr::Result& result : ocr.Recognize(ppocr::LoadPPM(argv[4]))) {
    std::cout << result.text << '\t' << result.confidence << '\n';
  }
}
```

```bash
./ppocr_cpp_api_demo det.onnx rec.onnx dict.txt image.ppm
```

Images passed to the library are packed RGB, row-major, 8-bit. `LoadPPM` reads binary P6. JPEG and PNG decoding stays in the application (Pillow does it for Python).

#### C

```c
#include "ppocr/ppocr.h"

#include <stdio.h>

int main(int argc, char** argv) {
  ppocr_options options;
  ppocr_options_init(&options);
  options.backend = PPOCR_BACKEND_CPU;

  ppocr_ocr* ocr = ppocr_create(argv[1], argv[2], argv[3], &options);
  ppocr_image_buffer loaded = {0};
  if (ppocr_load_ppm(argv[4], &loaded) != PPOCR_OK) return 1;

  ppocr_image image = {loaded.width, loaded.height, loaded.rgb};
  ppocr_result result = {0};
  if (ppocr_recognize(ocr, &image, &result) != PPOCR_OK) {
    fprintf(stderr, "%s\n", ppocr_last_error());
    return 1;
  }
  for (size_t i = 0; i < result.count; ++i) {
    printf("%s\t%.4f\n", result.items[i].text, result.items[i].confidence);
  }
  ppocr_result_free(&result);
  ppocr_image_free(&loaded);
  ppocr_destroy(ocr);
  return 0;
}
```

`ppocr_recognize` on one handle is serialized. Create one handle per thread when pages should run together. `ppocr_options.struct_size` is filled by `ppocr_options_init` so later fields can be appended.

#### Vulkan

Wheels are CPU-only. A Vulkan-enabled build selects the device in the same place 3.x puts `use_cuda`:

```python
engine = RapidOCR(params={
    "Det.model_path": "PP-OCRv6_det_tiny.onnx",
    "Rec.model_path": "PP-OCRv6_rec_tiny.onnx",
    "Rec.rec_keys_path": "ppocrv6_tiny_dict.txt",
    "Global.use_cls": False,
    "EngineConfig.ppocr_cpp.use_vulkan": True,
    "EngineConfig.ppocr_cpp.device_index": 0,
})
result = engine("hello.png")
print(result.txts)
```

`device_index` `-1` skips CPU Vulkan devices such as lavapipe. Set the index (or `PPOCR_VULKAN_DEVICE_INDEX`) when lavapipe is the device you want, and point `VK_ICD_FILENAMES` at its ICD. If no compute device exists, Python logs a warning and stays on CPU. `backend` accepts `cpu`, `hybrid`, or `vulkan`.

In C++ the default `Backend::hybrid` stays on CPU unless `PPOCR_ENABLE_HYBRID_GPU_GRAPH` is set. `Backend::gpu_only` throws `std::runtime_error` when the device or a shader cannot run. The C ABI returns `PPOCR_ERR_UNSUPPORTED` from `ppocr_create` in that case.

```cpp
ppocr::Options options;
options.backend = ppocr::Backend::gpu_only;
options.vulkan_device_index = 0;  // -1: automatic; CPU/lavapipe are skipped
ppocr::OCR ocr(det, rec, dict, options);
```

One recorded command buffer can drop CTC timesteps on some adapters. The default GPU path submits fenced segments. `PPOCR_ENABLE_GPU_GRAPH_REPLAY=1` opts into the single command buffer after that adapter has been qualified.

### ⚙️ Configuration

Python keys use the 3.x `Section.name` form. A YAML `config_path` is loaded with PyYAML and then overridden by `params`.

| Key | Role |
| --- | --- |
| `Det.model_path`, `Rec.model_path`, `Rec.rec_keys_path` | Local ONNX files and the recognizer dictionary. Omitted paths are downloaded. |
| `Det.model_type`, `Rec.model_type` | `tiny`, `small`, or `medium` inside the PP-OCRv6 URL table. |
| `Det.ocr_version`, `Rec.ocr_version` | `PP-OCRv6`. Older version names raise `NotImplementedError`. |
| `Det.limit_side_len`, `Det.limit_type` | Resize limit. `min` matches 3.x (short side). `max` is the C++ long-side default. |
| `Det.thresh`, `Det.box_thresh`, `Det.unclip_ratio` | Detector mask threshold, box score, and unclip. |
| `Det.mean`, `Det.std` | Detector normalization, BGR order, length 3. |
| `Det.use_dilation` | 2×2 dilation of the detector mask. On in the Python config, off in `ppocr::Options`. |
| `Global.text_score` | Drop lines below this score. Also a per-call argument. |
| `Global.use_cls` | Accepted. The native engine does not run an angle classifier and warns once. |
| `Global.model_root_dir` | Download directory. The package `models/` directory when null. |
| `Rec.rec_batch_num` | Recognizer batch size. |
| `EngineConfig.ppocr_cpp.use_vulkan` | Request the Vulkan graph. |
| `EngineConfig.ppocr_cpp.device_index` | Vulkan device. `-1` skips lavapipe. |
| `EngineConfig.ppocr_cpp.backend` | `cpu`, `hybrid`, or `vulkan`. |

C++ defaults live on `ppocr::Options` in `include/ppocr/ppocr.hpp` (`det_limit_side_len` 960, `det_limit_type` 0, ImageNet `det_mean` / `det_std`, `rec_batch_size` 4). `PPOCR_BENCH_THREADS` overrides the persistent SIMD pool (default `min(16, hardware_concurrency)`).

### 🔀 Migrating from rapidocr 3.x

`RapidOCR(config_path=..., params={"Section.key": value})` and

`engine(img, use_det=..., use_cls=..., use_rec=..., text_score=..., box_thresh=..., unclip_ratio=...)`

keep the same names. `result.txts`, `result.scores`, `result.boxes`, `result.elapse`, `to_json()`, `to_markdown()`, and `vis()` are still there.

| 3.x (`main`, through `v3.9.2`) | rapidocr 5 |
| --- | --- |
| ONNX Runtime, OpenVINO, Paddle, PyTorch, TensorRT, MNN | Built-in PP-OCRv6 interpreter. `EngineType.ONNXRUNTIME` and `EngineType.PPOCR_CPP` both select it. Other engine names raise `NotImplementedError`. |
| PP-OCRv4 and PP-OCRv5 model zoo | PP-OCRv6 ONNX detector and recognizer. |
| Angle classifier (`use_cls=True` by default) | The cls file is still downloaded when `use_cls` is true. The native engine ignores it and logs one warning. |
| `use_det=False` or `use_rec=False` | Raises `RapidOCRError`. Detection and recognition always run together. |
| Word and character boxes | `return_word_box` / `return_single_char_box` warn. `word_results` is empty. |
| `elapse_list` is `[det, cls, rec]` | `[None, None, total]`. An empty page is `RapidOCROutput()` (`boxes` / `txts` / `img` are `None`). |
| `vis()` draws with OpenCV in BGR | `vis()` draws with Pillow and returns RGB. |
| `EngineConfig` CUDA, OpenVINO, TensorRT, MNN, … | Those keys load and can be read. They are not applied. Vulkan keys above are applied. |

A caller that only reads `txts`, `scores`, and `boxes` from a PP-OCRv6 model can switch by installing this package. Set `use_cls=False` to skip the warning. Do not pass `use_det=False` or an older OCR version.

There is no 4.x tag in this repository. Details and the C ABI are in [`docs/API.md`](docs/API.md) and [`python/README.md`](python/README.md).

### ⚠️ Known limits

- No direction classifier. `use_cls` does not rotate boxes.
- No detection-only or recognition-only call. `use_det=False` and `use_rec=False` raise `RapidOCRError`.
- No per-word or per-character boxes. `word_results` stays empty.
- The native engine runs PP-OCRv6 ONNX graphs. PP-OCRv4 / v5 files are not executed.
- GPU-only mode has no CPU neural fallback. Hybrid stays on CPU unless `PPOCR_ENABLE_HYBRID_GPU_GRAPH` is set.
- Published wheels do not contain Vulkan shaders.

### 📚 Documentation

- [C / C++ / Python API](docs/API.md)
- [CPU performance notes](docs/PERFORMANCE.md) (kernel history; the table above is the later rerun)
- [Python package notes](python/README.md)
- [3.x documentation site](https://rapidai.github.io/RapidOCRDocs/) (Chinese, `main`)

### 📦 Releasing

Wheel and sdist builds run from [`.github/workflows/build-wheels.yml`](.github/workflows/build-wheels.yml) on pushes to `V5_NG`. PyPI publishing and the GitHub Release run only when a `v5*` tag is pushed. The steps, the trusted-publisher settings, and the tag command are in [`docs/RELEASING.md`](docs/RELEASING.md).

### 👥 Who use? ([more](https://github.com/RapidAI/RapidOCR/network/dependents))

Projects that depend on the `rapidocr` package include [Docling](https://github.com/DS4SD/docling), [CnOCR](https://github.com/breezedeus/CnOCR), [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR), and [LangChain](https://github.com/langchain-ai/langchain). The 3.x line on `main` is what those releases install today. Register a new dependent at [discussion 286](https://github.com/RapidAI/RapidOCR/discussions/286).

### 🙏 Acknowledgements

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for the models and the OCR pipeline this engine implements.
- [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch) and [PaddleX](https://github.com/PaddlePaddle/PaddleX) for the conversions and document models used by the wider RapidOCR project.
- Everyone listed on the [contributors](https://github.com/RapidAI/RapidOCR/graphs/contributors) graph.

<p align="left">
  <a href="https://github.com/RapidAI/RapidOCR/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=RapidAI/RapidOCR&max=400&columns=10" width="60%"/>
  </a>
</p>

### 🌟 Sponsors & Backers

RapidOCR is an Apache-2.0 project. Sponsorship is described on the [3.x docs site](https://rapidai.github.io/RapidOCRDocs/main/sponsor/).

| Sponsors | Application | Introduction |
| :---: | :---: | --- |
| <img src="https://github.com/RapidAI/RapidOCR/releases/download/v1.1.0/Quicker.jpg" width="65" height="65" alt="Quicker"/> | [Quicker](https://getquicker.net/) | Your fingertip toolbox |

### 📜 Citation

```bibtex
@misc{RapidOCR 2021,
    title={{Rapid OCR}: OCR Toolbox},
    author={RapidAI Team},
    howpublished = {\url{https://github.com/RapidAI/RapidOCR}},
    year={2021}
}
```

### ⭐️ Stargazers over time

![Star History](https://raw.githubusercontent.com/RapidAI/RapidOCR/star-tracker-data/charts/star-history.svg)

### ⚖️ License

RapidOCR source code is copyright RapidOCR Authors and licensed under the Apache License, Version 2.0 (`license = "Apache-2.0"` in [`pyproject.toml`](pyproject.toml) and [`python/pyproject.toml`](python/pyproject.toml)). The license text is at [apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0).

The OCR model files are derived from official PaddleOCR models. Copyright in the upstream weights is held by Baidu and/or the respective PaddleOCR rights holders. Those upstream models are distributed under the Apache License, Version 2.0. RapidOCR does not claim ownership of the upstream weights.
