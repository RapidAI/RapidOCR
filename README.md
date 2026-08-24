# Pure C++ PP-OCRv6 inference

This is a focused C++ port of the PP-OCRv6 detector/recognizer path from
`D:\workprj\aicoder\corelib\ocr`.  It has no dependency on ONNX Runtime,
Paddle Inference, OpenCV, or Python.

The embedded executor is intentionally not a general ONNX Runtime.  It only
implements the operator set exercised by the official PP-OCRv6 tiny/small/
medium ONNX model bundle: convolutions (including transpose), pooling,
elementwise math, normalisation, resize, tensor-shape operators, attention
matmul, and softmax.

> **Current GPU-only qualification (2026-08-21).** The strict Vulkan
> GPU-only OCR path runs RGB preprocessing, the complete neural graph, DB
> postprocessing, and CTC Top-1 on the GPU; it has no CPU neural-operator
> fallback. On the local AMD Radeon 780M / LLPC stack, exact CPU/GPU decoded
> text and geometry are currently verified for PP-OCRv6 **tiny** and **small**.
> GPU recognizer crops
> now retain their exact PP-OCR aspect-ratio width rather than adding a
> 64-column scheduling suffix, fixing short-crop CTC parity. The **medium**
> detector at production `1x3x160x704` remains unqualified and fails closed;
> no result is substituted from CPU. Nearby `1x3x160x640` and `1x3x160x768`
> shapes remain CPU/GPU parity-qualified. Historical GPU claims below do not
> widen this current qualification.

### 2026-08-22 strict GPU-only public-boundary recheck

The public `gpu_only` mode remains a strict Vulkan-only execution path:
packed RGB upload and resize, detector and recognizer graphs, DB box
postprocessing, and CTC Top-1 execute on the device. Failed Vulkan execution
does not retry neural inference on CPU.

This recheck also carries the Vulkan queue/fence status through every public
GPU-only OCR stage (RGB front end, detector, DB postprocess, recognizer and
CTC). A device reset is therefore reported as `VK_ERROR_DEVICE_LOST`, not
mislabelled as an unsupported ONNX operator. The only retry allowed by the
GPU-only page batch scheduler is an N=1 Vulkan retry after a rejected N-batch;
it never invokes the CPU executor. On the local Radeon 780M / LLPC adapter,
production medium-detector `1x3x160x704` still fails closed with that result.
Changing command-segment size, forcing queue-idle, using a fresh command
buffer, using the isolated terminal-depthwise shader, or using its
generic/sliced variant did not make that shape stable. This is a driver
compatibility limit, not a CPU-fallback opportunity.

### 2026-08-22 GPU-only deployment coverage

`gpu_only` is a complete execution mode, not a request to accelerate only
selected layers: packed RGB upload/resize, detector, DB postprocess,
recognizer, CTC Top-1, and the real NCHW page/crop batch paths stay on one
Vulkan queue.  Only final compact OCR records and UTF-8 construction reach
the host.  A rejected N-batch is retried as strict Vulkan N=1; it is never
redirected to the CPU graph.

The Vulkan loader is dynamically opened at runtime, so applications do not
link to Vulkan or ONNX Runtime.  The build now embeds shaders with the
vendored Windows compiler when available, otherwise with a host
`glslangValidator`/`glslang`; the same executor is therefore buildable on
Windows x86/x64, Linux x86/x64, and Linux ARM/AArch64 when Vulkan headers and
a shader compiler are supplied.  If the loader, compute queue, shader build,
or a required graph node is unavailable, constructor/runtime validation fails
closed rather than selecting CPU inference.

After this portability change, the local Radeon 780M validation was rerun
with the rebuilt `build-gpu-only-ep` binaries:

| Check | Result |
| --- | --- |
| tiny GPU-only OCR, `en` + `mixed` + low contrast, 2 repeats | passed; 8 records; max confidence delta `2.50e-5` |
| small GPU-only OCR, `en` + `mixed` | passed; 5 records; max confidence delta `1.0e-6` |
| tiny two-page GPU batch, `en` x2, 1 warm-up + 5 runs | serial `93.109 ms`, batch `84.491 ms`, `1.102x`; text/geometry equivalent |
| tiny GPU memory ladder, 16px through 1920px pages then back to small | all requests completed; post-large small/en/mixed returned to the warmed `596,709,376 B` private-byte baseline |

The last memory sweep later rose by at most `9,392,128 B` after a noisy page
and finished `7,172,096 B` above that process allocator baseline.  This is a
bounded host allocator effect observed after the request; it is not reported
as GPU arena growth and does not relax the device-only inference contract.

### 2026-08-22 expanded GPU-only corpus recheck

The correct PP-OCRv6 common dictionary (`dict_ppocrv6.txt`, vocabulary
`18710`) was used for small/medium rather than the tiny dictionary.  The
latest strict GPU-only corpus covers low contrast, very wide/tall source
images, and a 1920×1080 screenshot in addition to short English/mixed text:

| Model | Images | Result |
| --- | --- | --- |
| tiny | `en`, `mixed`, low contrast, `3000×80`, `80×3000`, screenshot; 2 repeats | passed; 63 records; exact decoded text/geometry; max confidence delta `1.1662e-3` |
| small | same six images | passed; 63 records; exact decoded text/geometry; max confidence delta `2.265e-4` |
| medium | complete OCR at production detector shape | unavailable on this Radeon 780M/LLPC driver: strict `VK_ERROR_DEVICE_LOST`; no CPU substitution |

The independent neural-graph checks remain passed for tiny/small
`1×3×160×704` and medium `1×3×160×640`, with maximum CPU/GPU errors
`5.63276e-7`, `3.05939e-7`, and `7.56991e-8` respectively. Small-model memory
through the same small-to-large ladder returned to its warmed baseline after
the final screenshot (`1,020,276,736 B` private bytes); a low-contrast request
briefly added `14,577,664 B` in the process allocator and the final screenshot
was below baseline. `ui_dense_1920x1080` is deliberately excluded from the
qualified OCR corpus: it currently has one CPU/GPU decoded-result mismatch.
That failure is retained as a regression target rather than being masked by a
fallback or presented as a pass.

The rebuilt `build-gpu-only-ep` binaries passed the currently qualified
coverage. The graph timing measurements below use two warmed runs on the same
machine and include the current host/device graph boundary:

| Check | Result |
| --- | --- |
| tiny detector graph, `1x3x160x704` | passed; max error `5.63276e-7`; CPU/GPU `10.590 / 45.256 ms` |
| small detector graph, `1x3x160x704` | passed; max error `3.05939e-7`; CPU/GPU `18.676 / 66.362 ms` |
| medium detector graph, `1x3x160x640` | passed; max error `7.56991e-8`; CPU/GPU `80.312 / 272.255 ms` |
| GPU-only OCR tiny, `en` + `mixed` | passed; 5 records; max confidence delta `2.9e-6` |
| GPU-only OCR small, `en` | passed; 2 records; max confidence delta `8e-7` |
| true two-page GPU batch, tiny `en` x2 | serial-equivalent; `93.177 -> 90.993 ms` (`1.024x`) |
| true two-page GPU batch, small `en` x2 | serial-equivalent; `239.759 -> 231.595 ms` (`1.035x`) |

The medium detector memory ladder
`64x128 -> 96x256 -> 128x512 -> 160x640 -> 64x128` also passed. At every
phase the live allocation returned to the `61,946,548`-byte immutable
constants baseline; reusable capacity reached `201,457,664` bytes and is
intentionally retained for later requests. These results validate qualified
GPU-only execution and recovery behavior, but do not claim that this UMA GPU
is faster than the AVX-512 CPU or an ORT GPU provider.

The public GPU-only memory ladder also covered, in increasing source-size
order, `tiny_16x16`, `en`, `mixed`, `noisy_lowcontrast`, `wide_strip_3000x80`,
`tall_strip_80x3000`, `screenshot_1920x1080`, and `ui_dense_1920x1080`.
After the final large image, processing the next small image retained no extra
private bytes above the warmed large-page baseline. A later noisy page can
make the process allocator grow by about 12 MB for compact host result/control
storage; that allocation stayed bounded through the remaining large pages.
GPU replay command buffers and their pinned slots are now discarded whenever
the model uses the bounded detector/recognizer hand-off policy, so old replay
shapes cannot accumulate as device allocations.

### 2026-08-22 hybrid Vulkan admission correction

Hybrid terminal CTC GEMM now follows the documented policy exactly: for the
same model shape, CPU worker budget, Vulkan adapter, and Vulkan runtime
generation, it is admitted when a complete upload/compute/download boundary
is **no slower than CPU**. The older `GPU <= CPU / 2` rule could reject an
equal-speed GPU path merely because multiple recognizer crops might execute
concurrently on CPU. Queue serialization is instead controlled at the OCR
scheduler boundary. The admission cache is now also tagged with the Vulkan
runtime generation, so a device-loss rebuild always re-measures rather than
reusing an answer from the old device.

On this Radeon 780M host, tiny `en` CTC terminal shapes measured `1.361 /`
`0.718 ms` and `1.846 / 1.281 ms` for GPU/CPU, so they correctly remain CPU
on this adapter. This is expected: the implementation enables Vulkan only on
actual no-slower results, and does not manufacture a GPU claim from a slower
boundary. The hybrid batch regression (`tiny_16x16`, `en`, `mixed`, low
contrast; 2-page scheduling) remained text/geometry equivalent and measured
`67.358 -> 63.918 ms` (`1.054x`).

### 2026-08-21 GPU-only requalification

#### Latest local recheck

The strict Vulkan build was rebuilt after removing an unproven automatic
generic-tail experiment. The standard terminal scalar depthwise selection is
restored for qualified shapes. The following tests were rerun on AMD Radeon
780M / LLPC; all use the pure C++ implementation without ONNX Runtime:

| Check | Result |
| --- | --- |
| tiny detector graph, `1x3x160x704` | passed, max error `5.63276e-7` |
| medium detector graph, `1x3x160x640` | passed, max error `7.56991e-8` |
| medium detector graph, `1x3x160x768` | passed, max error `3.25672e-8` |
| medium recognizer graph, `1x3x48x320` | passed, max error `2.66433e-5` |
| GPU-only OCR tiny, `en` + `mixed` | passed; 5 records; max confidence delta `2.9e-6` |
| GPU-only OCR small, `en` | passed; 2 records; max confidence delta `8e-7` |
| tiny graph memory ladder, `64x128 -> 96x256 -> 128x512 -> 160x704 -> 64x128` | passed; live device bytes always returned to `1,713,924` B |

The production medium detector at `1x3x160x704` still returns
`VK_ERROR_DEVICE_LOST` on this driver after the terminal `896x5x22` depthwise
stage. With the normal 20-node segmentation the failure is observed on the
segment ending at that stage; a diagnostic command-stream grouping can submit
that tail and subsequently fails in the FPN head. This identifies a
Radeon 780M/LLPC Vulkan command-stream stability issue rather than a usable
CPU fallback point. It remains deliberately fail-closed: no CPU neural or
postprocess substitute is used. This blocks a claim of complete all-model
GPU-only qualification on the local adapter.

The end-to-end GPU path was rebuilt from the conservative source and retested
on Radeon 780M/LLPC. A GPU crop may use a dynamic width; it is no longer
rounded to 64 columns, because padded input columns produce extra valid
CNN/CTC time rows and can alter a short crop's text or confidence.

| GPU-only OCR pair | Corpus | Result |
| --- | --- | --- |
| tiny | `tiny_16x16`, `en`, `noisy_lowcontrast`, `wide_strip_3000x80` | 7 results; exact text/geometry; max confidence error `2.50e-5` |
| tiny replay | `en` repeated 5 times on one live OCR instance | 10 results; exact text/geometry; max confidence error `4e-7` |
| small | `en`, `noisy_lowcontrast` | 5 results; exact text/geometry; max confidence error `1.3e-6` |

The strict graph sweep (three runs except medium detector, two) also passed:
tiny detector `9.367 / 28.870 ms` CPU/Vulkan, tiny recognizer `5.224 / 23.189
ms`, small detector `17.976 / 48.075 ms`, small recognizer `13.320 / 62.323
ms`, medium recognizer `30.569 / 123.212 ms`, and the qualified medium
detector shape `1x3x128x704` `78.976 / 234.478 ms`. These are full graph
latencies on this UMA GPU, not an assertion that it beats the AVX-512 CPU.
Maximum graph-output errors were `5.63e-7`, `2.18e-5`, `3.06e-7`, `3.78e-4`,
`2.66e-5`, and `6.71e-9` respectively.

GPU arena recovery was also rerun after the correction. In every small-to-
large-to-small phase, dynamic `activation_bytes=0` after completion:

| Graph ladder | Immutable live baseline | Final reusable capacity |
| --- | ---: | ---: |
| tiny detector through `1x3x160x704` | 1,713,924 B | 47,513,600 B |
| small detector through `1x3x160x704` | 9,813,844 B | 75,235,328 B |
| medium recognizer through `2x3x48x320` | 76,456,140 B | 174,850,048 B |

### 2026-08-21 verification update

The strict GPU graph replay policy was corrected for public host-input
`Run()` calls.  Replay remains enabled for device-input and in-graph RGB
execution, while host-input calls now release their dynamic activation slots
when the submission completes.  This is an arena-lifetime change only: every
neural operator remains a Vulkan dispatch and an unsupported or failed device
operation still fails closed.

The reproducible tiny-detector ladder
`1x3x64x128 -> 1x3x96x256 -> 1x3x128x512 -> 1x3x160x704 -> 1x3x64x128`
now passes `ppocr_gpu_graph_mem_soak`.  Its immutable live baseline is
**1,713,924 bytes**; after every sample, including the largest one, live
bytes return exactly to that baseline.  Reusable arena capacity grows to
**47,513,600 bytes** at the largest input and intentionally stays available
for the final small input, so capacity is not misreported as a live-memory
leak.

The same live-memory invariant also passed for the qualified medium-detector
ladder ending at `1x3x160x640`: its **61,946,548-byte** immutable baseline
was recovered after each sample, with a stable **216,072,192-byte** reusable
high-water capacity.

The medium detector was rechecked at the production `1x3x160x704` input.
On this Radeon 780M/LLPC driver it still fails closed with
`VK_ERROR_DEVICE_LOST` in the terminal `896x5x22` depthwise dispatch. The
dedicated scalar shader is now selected automatically for that exact Vulkan
operator (use `PPOCR_GPU_TAIL_GENERIC=1` only for diagnostics); it also fails
closed on this production shape, as did device-local and host-cached arena
variants. The nearby qualified `1x3x160x640` and `1x3x160x768` shapes pass
CPU/Vulkan output comparison, which further confirms this is a driver/shape
interaction rather than a CPU fallback.

Hybrid RGB front ends are now admission-driven by default: detector RGB+stem
and recognizer RGB+Conv.0 run only after their complete upload/dispatch/
readback boundary is both numerically validated and no slower than the native
SIMD path. `PPOCR_DISABLE_VULKAN_RGB_STEM` and
`PPOCR_DISABLE_VULKAN_REC_RGB_CONV` disable those probes for diagnostics; no
enable-only switch is required.

The default strict-graph command segment is now **20 nodes** (override with
`PPOCR_GPU_ONLY_SEGMENT_NODES`).  A fresh three-run Vulkan sweep selected that
setting for the Radeon 780M: detector graph means were tiny `30.201 ms`,
small `46.590 ms`, and qualified medium `239.388 ms`; recognizer means at
`1x3x48x320` were tiny `23.169 ms`, small `60.031 ms`, and medium
`123.182 ms`.  CPU/Vulkan maximum output errors remained respectively
`5.63e-7`, `3.06e-7`, `6.71e-9`, `2.18e-5`, `3.78e-4`, and `2.66e-5`.
This is a segment-lifetime optimization, not an assertion that the local UMA
GPU is faster than the AVX-512 CPU.

Memory recovery was also exercised across detector resolution and recognizer
width/batch diversity.  Small detector `64x128 -> 96x256 -> 128x512 ->
160x704 -> 64x128` always returned to its **9,813,844-byte** constants
baseline.  The medium recognizer ladder `1x3x48x64 -> 1x3x48x160 ->
1x3x48x320 -> 2x3x48x320 -> 1x3x48x64` always returned to its
**76,456,140-byte** constants baseline.  Capacity is intentionally retained
as reusable GPU storage; no dynamic activation remains live after a sample.

The CPU batch executor now supports real same-shape detector NCHW batches
through both fused `MaxPool -> Concat -> Conv` and `Concat -> Conv` nodes.
Those nodes now advance each source by its own NCHW channel-plane stride, so
unequal concat branches cannot cross-contaminate pages. The varied-size
four-image detector-batch regression passes for tiny/small/medium: decoded
text and geometry are exact, while FP32 confidence is checked against its
documented bounded tolerance. `rec_batch_size` now defaults to `1` because
this PP-OCRv6 transformer export can change CTC results for N>1 recognition
crops; deployments may still opt in and qualify it independently. GPU-only
retains its separate device-resident crop batching path.

The full small-to-large-to-small memory ladder (19 images, including
`5120x2880`, `7680x4320`, wide and tall aspect ratios) also passed for every
CPU model.  Per-resolution private-byte and working-set evidence is retained
in `build-gpu-only-final/memory_ladder_{tiny,small,medium}_cpu_20260821.json`.
In each run both counters fell below the post-8K high-water during the
recovery tail; final-small private memory was within **-0.05% / +0.33% /
+1.06%** of the warmed tiny/small/medium baselines respectively.

The current five-image CPU comparison is recorded in
`build-gpu-only-final/cpu_diverse_benchmark_20260821.json`. On this local
AVX-512 host (16 threads, two warmups, eight runs per case), native C++ beat
the installed ONNX Runtime 1.28.0 CPU provider on every case:

| PP-OCRv6 model | weighted latency improvement | weakest case |
| --- | ---: | ---: |
| tiny | 73.54% | 63.33% |
| small | 46.16% | 36.22% |
| medium | 9.22% | 1.45% |

This is a local CPU-only result. The installed ORT package lists
`AzureExecutionProvider` and `CPUExecutionProvider`; its observed variance,
especially on small inputs, means these figures are not a portable GPU or
architecture-wide claim.

## CPU acceleration and architecture support

The inference path remains dependency-free and portable across x86/x64 and
ARM/AArch64. Hot contiguous elementwise paths (Add/Sub/Mul/Div, ReLU,
HardSwish, and affine scale/shift),
pointwise `1x1` convolutions, ordinary/depthwise `3x3`/`5x5` convolutions,
spatial mean reduction, fused BatchNorm+GELU, and attention `MatMul` use
fused vector kernels:

- x86/x64: AVX-512F/FMA when supported by the running CPU and OS; otherwise
  AVX2/FMA; otherwise scalar code. Advanced ISA code is compiled into separate
  translation units and selected only after runtime CPUID/OS-state checks.
- ARM/AArch64: NEON when the compiler/target exposes it, with a scalar
  fallback otherwise.
- Larger GEMMs are divided over CPU cores; small GEMMs stay single-threaded to
  avoid thread-launch overhead. Direct convolutions use a persistent worker set
  (rather than creating threads per operator), defaulting to at most 16 workers;
  `PPOCR_BENCH_THREADS` is an explicit override for deployment tuning and A/B
  tests.
- PPOCR_DISABLE_AVX512=1 is an optional per-process A/B switch for x86
  deployments. It keeps all runtime safety checks and selects AVX2/FMA when a
  particular AVX-512 CPU or workload performs better at its non-AVX-512 clock.
- `ppocr_kernel_smoke` verifies paired/odd GEMM rows, scalar/vector column
  tails, optional bias, and a recognizer-sized projection. Run it normally and
  with `PPOCR_DISABLE_AVX512=1` to cover the dynamic AVX-512 and AVX2 paths.
- PPOCR_DISABLE_AVX512_POINTWISE8=1 is a narrower A/B switch for the AVX-512
  eight-output `1x1` register tile. It retains AVX-512 and its four-output
  pointwise kernel, so deployments can isolate this tile's cache/register
  trade-off without changing the baseline ISA. A current three-model A/B
  confirmed that this decision is shape-sensitive: do not globally force the
  wider tile merely because AVX-512 is available.
- PPOCR_ENABLE_AVX512_POINTWISE_RELU8=1 opt-in enables the analogous
  eight-output `1x1 Conv+ReLU` tile for deployment-local testing. The stable
  four-output fused-ReLU path remains the default because the wider tile's
  end-to-end result is workload/cache sensitive.
- Large stride-one `3x3` feature maps can likewise use an AVX-512 eight-output
  channel tile. It shares each input vector across eight output reductions;
  the dispatcher restricts it to maps of at least 16,384 pixels and falls back
  to the four-output tile for smaller layers, odd channel tails, AVX2, NEON,
  and scalar targets. `PPOCR_DISABLE_AVX512_CONV3X3_TILE8=1` keeps the
  four-output AVX-512 path for deployment A/B. The low-level regression now
  checks both an eight-channel tile and a nine-channel ReLU tail, which also
  caught and fixed the prior un-clamped SIMD tail-channel bug.
- AVX-512 also has a dedicated unit-stride `2x2 SAME_UPPER` convolution path
  for the detector reconstruction stem. Its large valid interior uses four
  explicit FMA loads per input channel; only the final row/column runs the
  ONNX zero-padding boundary formula. It is dynamically selected only on a
  supported AVX-512 CPU, preserves the AVX2/scalar fallback, and can be A/B
  disabled with `PPOCR_DISABLE_AVX512_CONV2X2_SAME=1`.
- Equal-shape residual `1x1 Conv -> Add` pairs are fused at graph load and,
  on a runtime-eligible AVX-512, AVX2, or NEON CPU, keep the convolution
  accumulators live until the final residual add/store. This removes the
  intermediate Conv activation read/write. The executor validates the
  residual's runtime NCHW shape before selecting that kernel; scalar,
  broadcast, and unexpected-shape cases retain the existing exact
  Conv-then-Add path.
- Post-folding `Conv -> Sigmoid` and `Conv -> HardSigmoid` gate pairs are
  also fused. Their convolution result is activated in place, removing the
  generic unary output allocation; HardSigmoid uses one exact SIMD pass for
  `clamp(alpha*x+beta, 0, 1)` on AVX-512, AVX2, and NEON.
- Direct `Conv -> Add([1,C,1,1])` channel biases are folded into immutable
  convolution biases at graph load. If the Add is followed by ReLU, the three
  operations become one fused Conv+ReLU node. This covers the recognizer stem
  and avoids a full NCHW broadcast traversal and activation allocation.
- Recognition `Conv -> Sigmoid -> Mul` Swish gates are recognized after
  Conv+BatchNorm folding. The fused node writes `x * sigmoid(x)` directly into
  the Conv destination, eliminating the two branch activation tensors while
  retaining the exact FP32 operation order. The standalone BatchNorm variant
  follows the same rule when a model cannot fold the affine layer.
- `PPOCR_APPROX_GELU=1` is an opt-in PP-OCRv6 deployment mode for eligible
  model-aware fused BatchNorm+GELU planes. Its AVX-512/AVX2 dispatch is based
  on per-plane shape rather than total batch size, so batching cannot change a
  page's GELU implementation; generic GELU retains exact ONNX Erf semantics.

Model constants are no longer copied for every `Run()` invocation: they stay
in the immutable model store and execution allocates only inputs and temporary
tensors.  This substantially lowers peak transient memory usage for the
detector/recognizer.

Example cross builds (use the appropriate compiler/toolchain installed by your
environment):

```powershell
# Native ARM64 build (Visual Studio developer prompt)
cmake -S . -B build-arm64 -A ARM64
cmake --build build-arm64 --config Release

# Cross build using an ARM64 CMake toolchain file
cmake -S . -B build-aarch64 -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64.cmake
cmake --build build-aarch64 --parallel
```

The generic executor is not compiled for a host-specific ISA, avoiding
accidental illegal instructions on older x86 CPUs and on ARM targets. The
AVX-512 path remains the default on a supported CPU, while the switch is kept
for deployment-specific validation.

## ONNX Runtime comparison benchmark

The repository contains a repeatable CPU-only benchmark. It compares the
complete detector + recognition pipeline on the same PPM image, uses ONNX
Runtime's `CPUExecutionProvider`, warms up both engines, and reports mean,
median, p95, load latency and throughput.

```powershell
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build --parallel
powershell -ExecutionPolicy Bypass -File .\bench\compare.ps1 -Runs 20 -Warmup 3
```

### 2026-08-14: wider correctness corpus and normalization A/B

The regression runner now accepts a UTF-8-BOM JSON case manifest as well as
plain UTF-8. This makes a small PowerShell-authored subset reproducible rather
than failing before inference. The current native validation covered all three
models against ONNX Runtime text output on English, Chinese, and `3000x80`
wide text (`3/3` cases each):
`build/accuracy_default_preprocess_ab_{tiny,small,medium}_20260814.json`.

Stress coverage also includes dense UI, tall text, and a `2560x1440`
screenshot. Tiny completed all 11 built-in cases; focused high-cost samples
completed for small (`3/3`) and medium (`2/2`), including 38 medium screenshot
regions. Reports are under `build/stress_*_current_20260814.json`.

Preprocessing retains its exact fused bilinear + normalization path. A compact
256-entry uint8 normalization lookup implementation was evaluated, but it
regressed medium on this AVX-512 host due to indexed-load/cache pressure. It
therefore remains an opt-in A/B switch (`PPOCR_ENABLE_PREPROCESS_NORMALIZE_LUT=1`)
rather than changing the production default. The paired result is recorded in
`build/bench_preprocess_lut_ab_current_20260814.json`; this keeps the default
chosen by measured end-to-end performance, not by a microbenchmark assumption.

The runtime AVX-512 dispatch remains dynamic: its current 15-run English A/B
against the AVX2 fallback measured **29.520 vs 34.133 ms** (tiny), **56.578 vs
80.106 ms** (small), and **212.366 vs 323.044 ms** (medium). See
`build/bench_avx512_dynamic_current_20260814.json`. Vulkan smoke continues to
pass on AMD Radeon 780M and admits the channel-affine+Swish and depthwise+Swish
GPU segments only when their complete transfer/dispatch/readback timing is no
slower than CPU. GPU-only correctly refuses construction until the complete
PP-OCRv6 device graph exists; it never silently runs CPU work.

`compare.ps1` now treats the requested performance target as a gate. Its
`-MinImprovementPercent` parameter defaults to `30`, and it returns a non-zero
exit code when end-to-end native latency is not at least that much lower than
the paired ONNX Runtime CPU measurement. Set it explicitly when a deployment
uses a different threshold.

`bench/ort_bench.py` requires `onnxruntime`, `numpy`, and Pillow in the Python
environment chosen by `bench/compare.ps1`. `ppocr_bench` keeps the native OCR
instance alive during all warmup/measurement iterations, so model parsing and
weight loading are excluded from steady-state latency just as they are in the
ONNX Runtime side.

### Latest measured result

The latest comparison is two consecutive `bench/compare.ps1` runs (tiny
det+rec, `build/regression_ppm/en.ppm`, `PPOCR_BACKEND=cpu`, 16 threads, 3
warmup / 20 measured, `PPOCR_APPROX_GELU` unset) captured on 2026-08-20 after
fusing the inverted-residual `1x1 expand → exact GELU → 1x1 project → residual
Add` into one spatial-tiled AVX-512 kernel (10 detector + 7 recognizer sites),
plus the prior AVX-512 stem/FPN/GEMM path. Model loading is excluded from
steady-state time. Figures are copied from those two JSON blobs; they are not
averaged into a third number.

| Run | Engine | Mean latency | Throughput | C++ / ORT | Latency improvement |
|---|---|---:|---:|---:|---:|
| 1 | Pure C++ PP-OCRv6 | 20.972 ms | 47.682 FPS | 0.208x | 79.16% |
| 1 | ONNX Runtime 1.28 CPUExecutionProvider | 100.631 ms | 9.937 FPS | | |
| 2 | Pure C++ PP-OCRv6 | 20.973 ms | 47.681 FPS | 0.209x | 79.15% |
| 2 | ONNX Runtime 1.28 CPUExecutionProvider | 100.588 ms | 9.942 FPS | | |

Both runs used `faster_engine = ppocr_cpp`. Native median/p95 were
20.884/23.443 ms and 20.376/23.720 ms. Relative to this change's pre-change
native ceiling of 30.280 ms, the pair is 30.7% / 30.7% lower latency
(each mean ≤ 21.196 ms). This is a host-specific result, not a portable
performance guarantee; rerun on the deployment CPU before capacity planning.

The previous pair after AVX-512 stride-2 tiles / fused MaxPool+Concat / FPN
Concat+Conv was 24.721 / 24.887 ms native vs 125.828 / 102.010 ms ONNX Runtime
(80.35% / 75.60%). The AVX2 exact-Erf GELU pair was 26.397 / 26.357 ms vs
99.273 / 111.544 ms (73.41% / 76.37%). The longer 60-run snapshot (2026-08-11,
8 warmup) on `build/en.ppm` was 33.074 ms native vs 138.359 ms ONNX Runtime.
Keep those as historical context only; the table above is the current
`compare.ps1` pair.

Reproduce the benchmark with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\bench\compare.ps1 `
  -Runs 20 -Warmup 3 -Threads 16 -Backend cpu `
  -NativeExe .\build-gpu-only-final\ppocr_bench.exe
```

`bench/compare.ps1` drives the tiny pair. For small/medium, invoke
`build/ppocr_bench.exe` and `bench/ort_bench.py` directly with their matching
models and dictionary, as shown by the commands below.

### Current three-model CPU comparison

After the ARM/NEON GEMM bias-broadcast correction, the current x86 AVX-512
build was rerun on `build/en.ppm` with `PPOCR_BENCH_THREADS=16`, three warmups,
and ten timed runs per engine. The native and ONNX Runtime runs use the same
detector/recognizer pair, complete OCR pipeline, image, thread budget, and
warmup/run counts. This is a fresh shared-host snapshot rather than a portable
guarantee, but it confirms that all three supported PP-OCRv6 sizes remain well
ahead of ONNX Runtime CPUExecutionProvider:

| Model | Pure C++ mean | ORT mean | C++ / ORT | C++ latency reduction |
|---|---:|---:|---:|---:|
| tiny | 29.757 ms | 142.158 ms | 0.209x | 79.1% |
| small | 63.395 ms | 198.205 ms | 0.320x | 68.0% |
| medium | 191.198 ms | 374.134 ms | 0.511x | 48.9% |

The corresponding native throughput was 33.606, 15.774, and 5.230 FPS,
compared with ORT's 7.034, 5.045, and 2.673 FPS. Re-run the paired commands on
the deployment hardware before using these values for capacity planning.
The scalar/NEON GEMM fallback was also corrected to broadcast the bias by
output column (the same semantics as AVX-512/AVX2), rather than incorrectly
using one bias value for a whole output row. This is a correctness fix for
ARM targets and does not change the dynamically dispatched x86 measurements.

### 2026-08-13: refreshed long three-model CPU/ORT comparison

After the AVX-512 stride-one `3x3` eight-output tile was revalidated as the
default on all three model sizes, the complete `en.ppm` OCR pipeline was
rerun with 16 native/ORT CPU workers, five warmups, and 40 timed iterations
per engine.  This is the longer same-host measurement; it includes detector,
crop construction, recognizer, and decoding, but excludes model loading from
steady-state latency.  ORT used CPUExecutionProvider only.

| Model | Pure C++ mean | ORT mean | C++ / ORT | C++ latency reduction |
|---|---:|---:|---:|---:|
| tiny | 32.244 ms | 119.475 ms | 0.270x | 73.0% |
| small | 67.857 ms | 151.359 ms | 0.448x | 55.2% |
| medium | 220.768 ms | 365.267 ms | 0.604x | 39.6% |

All three exceed the repository's 30% paired CPU performance gate on this
host.  The benchmark samples remain machine-specific and ORT has wider timing
variation on this shared Windows host; full raw measurements are retained in
`build/compare_three_models_tile8_long_20260813.json`, so this table is not a
cross-platform guarantee.

### 2026-08-11: refreshed three-model exact CPU comparison

A fresh paired short run on `regression_ppm/en.ppm` used the same complete
OCR pipeline, matching 16-thread budgets, three warm-ups, and ten timed runs.
AVX-512 was dynamically selected on this x86 host; no approximation mode was
enabled.  The native and ORT measurements were executed separately on a shared
Windows machine, so these are reproducible evidence for this host閳ユ攺ot a
portable capacity promise.

| Model | Pure C++ mean | ORT mean | C++ / ORT | C++ latency reduction |
|---|---:|---:|---:|---:|
| tiny | 31.251 ms | 129.990 ms | 0.240x | 76.0% |
| small | 70.912 ms | 201.911 ms | 0.351x | 64.9% |
| medium | 199.734 ms | 409.505 ms | 0.488x | 51.2% |

This refresh keeps all three PP-OCRv6 sizes above the requested 30% native
CPU latency advantage on the same exact input. It follows the recognizer
executor lifetime fix that now executes `FusedBatchNormSwishAdd` in the
terminal-CTC path as well as the ordinary graph executor. The path reuses the
dying BatchNorm activation for affine + exact Swish + shortcut Add, reducing
the live feature-map frontier for small/medium crop batches. A profile confirms
the node as `ctc:FusedBatchNormSwishAdd:BatchNormalization.2_swish_add`.

The batch/memory follow-up used six varied PP-OCRv6 tiny inputs閳ユ攩rom `16x16`
through a `3000x80` strip閳ユ敋ith English, Chinese, and mixed content. Serial and
`RecognizeBatch()` outputs were identical (10 total results); bounded two-page
scheduling measured **134.218 ms serial / 86.501 ms batch (1.552x)**. A
medium-model memory cycle spanning `16x16`, mixed text, and a dense
`1920x1080` page used 369,672,192 private bytes after warm-up and finished at
+3,145,728 bytes. This is a Windows allocator high-water observation, not a
leak assertion; it bounds the varied-shape batch check to a small transient
increase rather than retaining every crop workspace.

## Vulkan backend status

The public API now exposes `ppocr::Backend::{cpu_only,gpu_only,hybrid}` and
`QueryBackendInfo()`. Runtime Vulkan probing is compiled from vendored Khronos
headers and dynamically loads `vulkan-1.dll`, so the binary does not require a
Vulkan SDK installation on its deployment machine. A generated SPIR-V compute
shader and smoke executable validate FP32 Add/Sub/Mul/Div building blocks over
scalar, non-workgroup-aligned, and recognizer-scale tensor sizes.
On this host, `ppocr_demo --backend-info` and `ppocr_vulkan_smoke` report the
AMD Radeon 780M compute queue and pass those correctness checks.

`gpu_only` deliberately fails if a complete PP-OCRv6 GPU graph is not
available; it never silently falls back to CPU. `hybrid` now connects verified
equal-shape in-place Add/Sub/Mul/Div executor nodes and eligible `1x1` Conv
nodes to Vulkan. For NCHW recognition tensors it maps N to the shader's Y
dimension, so a same-width crop batch uses one submission instead of one
submission per crop. A cached
per-shape/per-operation admission benchmark validates results and includes
host/device transfer, dispatch, and readback before it may select GPU; all
other operators remain on the portable CPU path. The Vulkan loader, instance,
device, queue, pipeline, descriptor set, command buffer, and grow-only
host-visible buffers are persistent, avoiding a per-dispatch Vulkan rebuild.

### 2026-08-12: depthwise GPU activation fusion

The batched Vulkan depthwise primitive now fuses its directly following
`ReLU` or exact `Swish` into the convolution writeback.  This covers the
common PP-OCRv6 MobileNet depthwise-gate shape without creating a CPU-side
activation pass or an intermediate host tensor.  The runtime still performs
shape-specific full-boundary admission (H2D, dispatch, fence, D2H, and CPU
SIMD comparison), and the new Vulkan smoke case validates batched odd-size
depthwise+Swish values against the scalar reference.  It is therefore a
batch-capable hybrid building block, not a claim that the still-partial GPU
graph is universally faster; `gpu_only` remains unavailable.

On the local Radeon 780M, the `N=4,C=32,H=31,W=29,3x3` admission sample was
**0.36344 ms GPU vs 0.62156 ms CPU**, so hybrid selected the fused GPU path.
The unfused depthwise sample in the same run was **0.33214 ms GPU vs 0.34084
ms CPU**. These figures include all transfers and synchronization and are
recorded as a per-segment comparison, not an end-to-end OCR claim.

The same depthwise kernel now also has a fused exact HardSwish variant for
PP-OCRv6 MobileNet blocks.  Its smoke admission sample measured **0.32760 ms
GPU vs 0.32076 ms CPU**, so hybrid correctly retained CPU for that shape;
the decision is dynamic rather than a blanket GPU preference.

### 2026-08-12: five-size batch and memory recheck

After the depthwise fusion work, CPU and hybrid `RecognizeBatch()` were
rechecked against serial execution on five deliberately different inputs:
`16x16`, English text, low-contrast text, a `3000x80` ultra-wide strip, and
CJK. All three official model sizes preserved exactly the same nested OCR
results. With one page worker, `det_batch_size=4`, and `rec_batch_size=4`, the
CPU measured serial/batch pairs were tiny **204.163/190.580 ms (1.071x)**,
small **392.496/375.710 ms (1.045x)**, and medium
**1260.420/1304.108 ms (0.967x)**. This mixed-shape result confirms that
batching is bounded-capacity scheduling rather than an unconditional latency
promise. The corresponding hybrid run was also result-identical (tiny
**204.311/210.308 ms**, small **436.332/411.353 ms**, medium
**1420.791/1461.834 ms**).

For compatible same-shape work, eight `en.ppm` pages with `det_batch_size=8`
and `rec_batch_size=8` measured **1.485x** CPU page-batch throughput for tiny
and **1.201x** for small. Medium did not benefit in that shared-host sample
(**0.911x**), and is therefore left with the conservative defaults rather
than promoting a batch size that regresses it. A two-cycle five-size tiny
memory soak started from **137,195,520 bytes** after warm-up and ended at
**+32,768** then **-151,552 bytes**, showing no retained per-shape activation
frontier.

Decoded-text comparison against ONNX Runtime was also rerun after these
changes for tiny, small, and medium on four diverse dimensions (`16x16`,
English, low-contrast, and `3000x80`): all **12/12** cases passed.

### 2026-08-12: GPU-aware batch regression refresh

The GPU and batch paths were rerun after the latest Vulkan/CPU optimization
pass. The public `RecognizeBatch()` API uses an exact-shape `[N,3,H,W]`
detector batch and same-width `[N,3,48,W]` recognizer crop batch; it never
rescales or pads unrelated page sizes merely to enlarge `N`. `det_batch_size`,
`rec_batch_size`, `rec_batch_width_bucket`, and `image_batch_parallelism`
remain the bounded controls, with `ppocr_batch_smoke` checking every batched
result against serial OCR before timing it.

On the local AVX-512 / Radeon 780M host, eight identical `en.ppm` tiny-model
pages (`det_batch_size=8`, `rec_batch_size=8`, one page worker) returned the
same 16 decoded results as serial execution. The measured CPU serial/batch
pair was **329.520 / 201.876 ms (1.632x)**; hybrid was **382.402 / 237.445 ms
(1.610x)**. A five-size small-model check (`16x16`, English, low contrast,
`3000x80`, CJK; batch sizes 4) likewise preserved all 9 results: CPU
**520.065 / 512.529 ms (1.015x)** and hybrid **438.718 / 427.016 ms
(1.027x)**. The mixed-size figures are intentionally not presented as a
universal throughput guarantee.

The refreshed Radeon Vulkan primitive suite passed numerical validation for
every existing batch primitive. Its strict full-boundary probes again admitted
the heavier fused work where it won (affine+Swish **0.16926 ms GPU vs 0.31296
ms CPU**; depthwise+Swish **0.36218 vs 0.66808 ms**; depthwise+HardSwish
**0.36482 vs 0.37636 ms**) and retained CPU for transfer-bound `1x1` Conv,
ordinary/transpose Conv, resize+add, and GEMM. This is intentional dynamic
selection: `gpu_only` remains explicitly unavailable until the complete
detector, recognizer, transformer, and CTC graph are device-resident.

### 2026-08-12: three-model CPU/GPU batch baseline refresh

The current pure-C++ build was rebuilt and rechecked on the complete
tiny/small/medium model set. The SIMD kernel suite passed under normal
AVX-512 dispatch and again with `PPOCR_DISABLE_AVX512=1`, exercising the AVX2
fallback. The Vulkan smoke suite passed on the Radeon 780M, including batched
binary/broadcast/affine/pointwise/depthwise/ordinary-convolution/transpose/
resize-add/GEMM primitives. In that run the full-boundary admission policy
selected Vulkan for depthwise (**0.35698 vs 0.36624 ms**), depthwise+Swish
(**0.35930 vs 0.59468 ms**), depthwise+HardSwish (**0.34864 vs 0.41902 ms**),
and affine+Swish (**0.17778 vs 0.34676 ms**); all transfer-bound primitives
stayed on CPU. The decision is shape-local and dynamic, so it lowers CPU work
without forcing a slower device boundary.

Decoded-text parity against ONNX Runtime passed **4/4** diverse inputs for
each model (tiny, small, medium): `16x16`, English, low contrast, and an
ultra-wide `3000x80` strip. A five-size serial-versus-batch gate for medium
also preserved all 9 results in CPU and hybrid modes. Its measured CPU pair
was **1257.378 / 1292.344 ms** and hybrid was **1266.158 / 1279.121 ms**;
this heterogeneous batch did not improve latency, so it is recorded as a
correctness/capacity check rather than a throughput claim.

For an `en.ppm` single-page steady-state snapshot (two warmups, five runs,
16 kernel workers), current CPU means were tiny **38.755 ms**, small
**73.192 ms**, and medium **208.209 ms**. Hybrid means were **34.575 ms**,
**75.141 ms**, and **238.014 ms** respectively. The short shared-host sample
therefore supports hybrid for tiny when the admitted GPU segments help, while
CPU remains the sensible choice for small/medium on this UMA device. These
numbers are host-specific and do not change the explicit `gpu_only` contract.

### 2026-08-12: hybrid admission re-audit

The next GPU-admission pass measured the current single-primitive boundaries
again instead of lowering thresholds blindly. On this Radeon 780M, batched
depthwise+Swish remained a useful GPU segment (**0.32756 ms GPU vs 0.57274 ms
CPU**), and the depthwise+HardSwish sample also crossed the no-slower rule
(**0.32540 vs 0.36794 ms**). Pointwise Conv, ordinary Conv, transpose Conv,
resize/add, and GEMM samples remained transfer/synchronization-bound, so they
continue on CPU. This preserves the hybrid contract: GPU is selected whenever
the complete measured boundary is equal or faster, not merely because a
shader exists.

The current source also retains conservative size guards before invoking an
admission microbenchmark. A short experiment removing them caused extra
small-shape probes and worse end-to-end tiny behavior on this UMA device, so
the guards were retained. Fresh post-audit checks passed kernel SIMD smoke,
the full Vulkan smoke suite, three-model four-size ONNX Runtime decoded-text
parity (**12/12**), five-size serial/batch output identity, and a medium
hybrid two-cycle memory soak. The latter began at **494,878,720 bytes** after
warm-up and ended **9,105,408** and **11,042,816 bytes below** baseline.

### 2026-08-12: GPU batch revalidation

GPU execution remains batch-aware: equal-width recognition crops are one NCHW
Vulkan dispatch (the shader maps crop `N` to its Y dimension), while the CPU
page scheduler bounds concurrent detector/recognizer workspaces. The pointwise
Conv shader stages compact immutable `[M,C]` weights in workgroup memory and
handles normal, ReLU, Swish, Sigmoid, and residual-Swish forms without an
intermediate activation. Hybrid admission still measures the entire H2D +
dispatch + D2H path for each shape and uses GPU only when the validated result
is no slower than the CPU SIMD counterpart.

### 2026-08-12: exact-shape detector batching

`RecognizeBatch()` now also coalesces pages whose normal PP-OCR detector
resize dimensions are exactly equal. This is a real `[N,3,H,W]` detector
batch: no document is padded, and each result is independently DB
postprocessed from its own probability-map slice. `Options::det_batch_size`
(default `4`) bounds the detector activation frontier; the smoke harness
accepts `PPOCR_DET_BATCH_SIZE` for A/B runs. CPU batch kernels therefore see
useful detector `N` work in addition to the existing same-width recognizer
crop batches.

The current hybrid path deliberately keeps this new detector batching at
`N=1`. Its Vulkan coverage is still partial and synchronous, so applying an
unproven large detector GPU segment would be transfer/admission-bound rather
than an optimization. CPU-only detector batches are available now; hybrid
continues to batch recognizer crops and only admits validated no-slower GPU
segments. This preserves the explicit `gpu_only` contract: full GPU OCR is
not claimed until detector, recognizer, transformer, and CTC stay
device-resident.

On this AVX-512 / Radeon 780M host, tiny PP-OCRv6 over eight identical
`en.ppm` pages (recognizer batch `8`, one worker) produced exactly the same
16 OCR results as serial inference. The steady-state smoke measurement was
**323.281 ms serial / 195.744 ms batch (1.652x)** with `PPOCR_DET_BATCH_SIZE=8`,
versus **330.019 / 247.684 ms (1.332x)** with detector batching forced to one.
The hybrid check preserved output identity at **311.225 / 243.242 ms
(1.279x)**; as designed, it retained singleton detector dispatches. A varied
five-page CPU run (`16x16`, English, low contrast, `3000x80`, and dense
`1920x1080`) stayed correct but measured only **1.019x**, so equal-shape
coalescing is a throughput tool for compatible page sets閳ユ攺ot a universal
mixed-document latency guarantee. Tiny, small, and medium each also retained
their four-image C++/ONNX Runtime decoded-text parity gate after the change.

### 2026-08-12: FPN resize-plus-add write fusion

The detector loader now recognizes the exported FPN sequence `Resize(nearest,
asymmetric floor, 2x) -> Add(equal-shape residual)` and replaces it with a
single `FusedNearestResizeAdd` node. Its NCHW implementation writes each
replicated source value directly into the final residual-sum allocation. This
removes the enlarged temporary feature map, its allocation, and its second
full read/write traversal. The fusion is deliberately limited to the exact
2x detector geometry; all other ONNX Resize combinations retain the generic
executor.

The local profile confirms all three recursive FPN additions are fused
(`Resize.0_add`, `Resize.1_add`, `Resize.2_add`). The post-change five-run
comparison on `en.ppm` (2 warm-ups, 16 benchmark threads) measured:

| Model | C++ hybrid mean | ORT CPU mean | C++ latency advantage |
|---|---:|---:|---:|
| tiny | 32.104 ms | 148.088 ms | 78.32% |
| small | 66.087 ms | 237.150 ms | 72.13% |
| medium | 259.850 ms | 454.815 ms | 42.87% |

This is one host's paired measurement rather than a general hardware claim.
Decoded-text parity against ONNX Runtime passed for each model on four varied
inputs (`16x16`, normal English, low contrast, and a `3000x80` strip). A
medium two-cycle varied-shape soak (`16x16`, English, low contrast, strip,
and `1920x1080` UI) ended at **-77,824 bytes** versus its warmed 391,249,920
byte private-memory baseline; this is allocator reuse evidence, not a claim
of zero memory use or a leak proof.

### 2026-08-12: runtime-SIMD FPN resize-add kernel

`FusedNearestResizeAdd` now calls a dedicated kernel rather than using scalar
replication inside the graph executor. The new kernel performs the 2x
replication and both residual-row additions in the same final stores, with
runtime AVX-512, AVX2, ARM NEON, and scalar fallback implementations. It is
covered by odd-width and batched low-level smoke cases, and
`PPOCR_DISABLE_RESIZE2X_ADD_SIMD=1` provides a narrowly scoped scalar A/B
path without disabling the other inference SIMD kernels.

On the local AVX-512 machine, a dense `1920x1080` tiny page (123 detected
crops, 2 warm-ups / 5 timed runs) measured **264.174 ms** with this SIMD
path versus **265.651 ms** through the scalar resize-add fallback (about
**0.56%** faster). This small but repeatable bandwidth-path improvement is
reported separately because whole-page timing remains sensitive to crop-count
and host load; correctness was also checked with AVX-512 disabled.

The refreshed local Radeon 780M smoke run passed all binary, broadcast,
affine, pointwise, residual, odd-plane, batched, and parameterized
`Conv+HardSigmoid` graph primitives. Its measured admission samples were:
affine+Swish **0.16466 ms GPU / 0.29544 ms CPU** (admitted), standalone `1x1`
Conv **0.14536 / 0.01952 ms**, and residual `1x1` Conv+Swish **0.18884 /
0.10648 ms** (both correctly retained on CPU). A tiny `en.ppm` pipeline check
(1 warm-up, 8 runs) measured **30.122 ms CPU** and **29.521 ms hybrid**; output
stayed identical. This small 2.0% difference is host-noise-sensitive and is
not presented as a general GPU speed claim.

### 2026-08-12: three-model, diverse-size regression

The CPU executor now has a dedicated `1x1 Conv -> HardSwish` path for the
canonical detector squeeze-excitation sequence. It writes convolution output
directly to the final activation allocation, then applies the existing
runtime-selected AVX-512/AVX2/NEON/scalar HardSwish kernel in place. This
preserves the exact `Conv -> HardSigmoid -> Mul` FP32 result while avoiding the
generic graph's temporary Conv activation. The Vulkan pointwise shader also
now covers parameterized `Conv+HardSigmoid`; its alpha/beta are passed as raw
FP32 push constants and remain behind the same complete-transfer admission
rule.

After the change, C++ versus ONNX Runtime decoded-text parity passed **7/7**
tiny samples covering English, CJK, blank, `16x16`, `3000x80`, low-contrast,
and mixed-content inputs. Small and medium also passed the ONNX Runtime text
gate on `en.ppm`. Kernel and Radeon 780M Vulkan smoke suites passed, including
the batched odd-plane pointwise and new HardSigmoid case.

On `en.ppm` (2 warm-ups, 10 timed runs, AVX-512 host), the measured
end-to-end comparison was:

| Model | C++ CPU mean | ONNX Runtime CPU mean | C++ advantage |
|---|---:|---:|---:|
| tiny | 29.159 ms | 121.267 ms | 4.16x faster |
| small | 61.785 ms | 184.410 ms | 2.98x faster |
| medium | 190.529 ms | 360.038 ms | 1.89x faster |

These measurements use the repository's ORT benchmark with the same image and
CPU execution provider; they are host-specific, but demonstrate a consistent
gain for all three PP-OCRv6 model sizes. A multi-size page-batch regression
(`16x16`, `3000x80`, and dense `1920x1080`) preserved output identity for
tiny/small/medium. Its throughput was 1.035x for tiny and 0.982x/0.975x for
small/medium on this mixed workload, so batching remains a bounded-capacity
feature rather than a universal latency claim. A medium two-cycle mixed-size
memory soak ended only **+4,345,856 bytes** above its 374,845,440-byte warmed
baseline.

### 2026-08-12: CTC LayerNorm activation reuse

`RunCtcTop1()` is the recognizer path used by public OCR calls. It now reuses
the input allocation for a fused transformer `LayerNorm` when that input has
one remaining consumer. LayerNorm first completes the mean/variance reductions
for a row and only then writes that row, so the source/destination alias is
safe and preserves the existing FP32 expression. This removes eligible
`[N,T,C]` temporary activations in the small and medium recognizers without
changing model scope or introducing a runtime dependency.

The feature has an A/B escape hatch, `PPOCR_DISABLE_CTC_LAYERNORM_INPLACE=1`,
for deployment validation. On the local AVX-512 host it produced modest
steady-state benefit on the short `en.ppm` workload (medium 225.182 ms enabled
versus 227.161 ms disabled, 2 warm-ups / 10 runs); actual benefit is expected
to be more useful for activation pressure on dense crop batches. The full
post-change regression passed kernel SIMD smoke, Radeon 780M Vulkan smoke,
the seven-image tiny ONNX Runtime text gate, and small/medium ONNX Runtime
text parity on `en.ppm`. The small/medium three-size batch gate also preserved
all 121/124 decoded results. `gpu_only` remains explicitly unavailable until
the entire detector, recognizer, transformer, and CTC pipeline is
device-resident; hybrid continues to admit each batch-shaped Vulkan segment
only when end-to-end transfer, dispatch, and readback are no slower than CPU.

### 2026-08-12: hybrid batch verification harness

`ppocr_batch_smoke` now accepts `PPOCR_BACKEND=cpu|hybrid|gpu` and the
recognizer batch controls `PPOCR_REC_BATCH_SIZE` and
`PPOCR_REC_PARALLELISM`. This makes the same serial-versus-page-batch
bit-for-bit output gate usable for the GPU-enabled hybrid path, instead of
testing only CPU scheduling. `gpu` retains the library's explicit failure
contract while the full PP-OCRv6 graph is not Vulkan-resident.

On the local AVX-512 / Radeon 780M host, tiny PP-OCRv6 with four varied pages
(`en`, `line_en`, `mixed`, `zh`), recognizer batch size 4 and one recognizer
worker produced eight identical OCR results in both serial and page-batched
modes. The measured scheduler results were:

| Backend | Serial | Page batch (parallelism 2) | Speedup |
|---|---:|---:|---:|
| CPU | 120.904 ms | 81.793 ms | 1.478x |
| Hybrid | 147.315 ms | 104.593 ms | 1.408x |

The hybrid result is intentionally reported as measured, not as a GPU win:
on this UMA GPU the available Vulkan segments are partial and transfer-bound
outside the admitted affine+Swish shape. The key correctness result is that
batch output remains identical while all runtime GPU selection stays
per-shape, batched, and no-slower-than-CPU. Reproduce with:

```powershell
$env:PPOCR_BACKEND='hybrid'
$env:PPOCR_REC_BATCH_SIZE='4'
$env:PPOCR_REC_PARALLELISM='1'
.\build\ppocr_batch_smoke.exe DET.onnx REC.onnx dict.txt 2 `
  .\build\regression_ppm\en.ppm .\build\regression_ppm\line_en.ppm `
  .\build\regression_ppm\mixed.ppm .\build\regression_ppm\zh.ppm
```

### Batch scheduling and current GPU evidence

`OCR::RecognizeBatch(const std::vector<Image>&)` is now the public page-batch
API. It preserves image/result order and exact per-image OCR output, while each
page keeps the existing same-width recognition-crop batching. Page images are
scheduled independently rather than padded into a single detector tensor: this
avoids wasted detector pixels and preserves PP-OCRv6's dynamic resize
semantics. `Options::image_batch_parallelism` bounds live page workspaces
(default `1`); when it is above one, the configured recognizer-worker budget
is divided among concurrently running pages so image batching does not multiply
the CPU thread count or activation frontier. `ppocr_batch_smoke` checks every
nested batch result against serial `Recognize()` before reporting timing.

On the local AVX-512 / Radeon 780M host, after warm-up, the tiny model over
four mixed PPM pages (`en`, `mixed`, `line_en`, `line_zh`) produced the same
seven OCR results in serial and batch modes. A short 2026-08-11 CPU snapshot
with page parallelism `2` measured **81.981 ms serial** and **53.183 ms batched**
(**1.541x** throughput). This is a scheduler/host-specific result, not a
promise for every page mix; the bounded default remains one page to minimize
peak memory. Reproduce it with:

```powershell
.\build\ppocr_batch_smoke.exe DET.onnx REC.onnx dict.txt 2 `
  .\build\regression_ppm\en.ppm .\build\regression_ppm\mixed.ppm `
  .\build\regression_ppm\line_en.ppm .\build\regression_ppm\line_zh.ppm
```

The Vulkan path already batches same-width NCHW crop tensors in one dispatch
and holds model-constant compact RHS/BatchNorm coefficients in persistent
buffers. The post-change Radeon 780M smoke run passed scalar, odd-sized,
four-stage chain, four-crop, broadcast, affine, and fused affine-Swish checks.
Its measured full-transfer admission samples were binary **0.29160 vs 0.01368
ms**, four-stage chain **0.16206 vs 0.02984 ms**, four-crop chain **0.14928 vs
0.02766 ms**, and channel-affine **0.13928 vs 0.01326 ms** (GPU vs CPU), so
hybrid correctly rejected those transfer-bound segments. The batched
affine-Swish segment measured **0.18272 ms GPU vs 0.30170 ms CPU** and was
therefore admitted for that exact shape. This is an isolated, verified GPU
segment; full `gpu_only` OCR remains unavailable until the convolution,
pooling, resize, transformer, and CTC graph is device-resident.
On the Radeon 780M, the latest 65,537-float repeated admission test measured
about **0.119 ms GPU versus 0.013 ms CPU**, so hybrid correctly declines this isolated
elementwise segment rather than increasing latency or CPU work. This is an
honest partial-graph capability contract, not a claim that GPU inference is
already complete.
The same persistent shader now supports a dependent chain of up to four
Add/Sub/Mul/Div operations in one dispatch, reusing device-resident operands
without intermediate host transfers. Its smoke test validates a 65,537-float
four-stage chain and applies the same full H2D/dispatch/D2H comparison. On this
Radeon 780M it measured about **0.124 ms GPU versus 0.030 ms CPU**, so hybrid again
correctly declines the GPU segment. This supplies a reusable fused-elementwise
building block for future PP-OCR graph segments without misrepresenting it as
complete GPU inference.
The Vulkan path also implements compact suffix-broadcast channel gates such as
`[N,C,H,W] op [N,C,1,1]`: the right operand is transferred once per batch
shape instead of being materialized to `N*C*H*W`. Runs whose four lanes stay
inside a channel span use `vec4`; only span boundaries use the scalar tail.
Its own full-transfer admission benchmark validates every output and retains
CPU SIMD unless the compact-GPU path wins. On this Radeon 780M, a four-item
8,192-float, 64-value-span `Mul+Add` gate measured **0.143 ms GPU versus
0.013 ms CPU**, so the policy correctly declined it. This is nevertheless
important GPU groundwork: it removes broadcast-expansion memory/traffic and
is connected to both generic and terminal-CTC in-place executor paths.

The batch path now also distinguishes the two valid compact RHS layouts:
dynamic `[N,C,1,1]`, and the much more common learned constant
`[1,C,1,1]`. The latter is now stored once in the persistent Vulkan RHS buffer
and addressed with a zero batch stride in the shader, rather than being
duplicated/uploaded once for every recognition crop. The Vulkan smoke test
executes two different four-crop activations against the same immutable
128-value RHS to verify both batch indexing and cache reuse. On the AMD
Radeon 780M, measured on 2026-08-11 after this change, the full-transfer
four-crop `8,192`-float `Mul+Add` shared-RHS building block was **0.11126 ms
GPU versus 0.00772 ms CPU**, so hybrid still (correctly) selects CPU SIMD.
This is a real batch-memory/traffic reduction, not a claim of end-to-end GPU
OCR acceleration while the surrounding graph remains CPU-resident.
The next batched GPU primitive is the exact inference affine
`x * scale + bias` used by fused BatchNorm. It keeps both `[1,C,1,1]`
coefficient vectors compact, caches immutable scale/bias pairs separately,
and transforms all same-width crops in one dispatch; it is wired into both
normal and terminal-CTC executor paths behind the same full-transfer admission
gate. The refreshed Radeon 780M smoke run (2026-08-11) measured the four-crop
`8,192`-float shared-coefficient affine at **0.10054 ms GPU versus 0.00656 ms
CPU**, so hybrid correctly retains the CPU SIMD affine path on this UMA device.
This establishes a useful batch/residency building block without pretending
that the present partial Vulkan graph makes end-to-end OCR faster.
The same compact four-crop kernel now also has an exact FP32
`BatchNorm affine -> Swish` mode, matching the recognizer's folded
`FusedBatchNormSwish` nodes. It evaluates `x * scale + bias` and its logistic
activation in one Vulkan dispatch, keeps immutable `[1,C,1,1]` scale/bias
vectors cached, and avoids returning to the CPU between the two operations.
It is connected to the hybrid executor through its own per-shape end-to-end
admission check. On the Radeon 780M on 2026-08-11, the four-crop
`8,192`-float shared-coefficient test measured **0.16744 ms GPU versus
0.37652 ms CPU**, so this heavier fused batch segment was admitted to GPU;
the smoke test validates every result against the exact CPU Swish expression.
This is a genuine batched hybrid acceleration, while the surrounding PP-OCR
graph remains CPU-resident and `gpu_only` remains unavailable.

### 2026-08-11: batched pointwise-Conv GPU experiment

The Vulkan executor now also contains a verified FP32 NCHW `1x1` Conv
primitive, with optional fused ReLU, Swish, and Sigmoid. It accepts a batch of equal-width
recognition crops in one dispatch, reads learned `[M,C]` weights and bias once
from persistent compact buffers, and writes to a separate lazily allocated
output buffer. The shader processes four adjacent spatial positions per
invocation when they remain in one output-channel plane; scalar tails preserve
odd widths and channel boundaries. The hybrid executor is wired only for
unit-stride, unpadded, ungrouped `Conv` / `Conv+ReLU` with immutable ONNX
weight+bias inputs. As with existing hybrid nodes, each `(N,C,M,plane,ReLU)`
shape is admitted only after an end-to-end H2D/dispatch/D2H benchmark validates
every FP32 result against the CPU SIMD kernel.

On the local AMD Radeon 780M, the four-crop `C=24, M=40, plane=257` smoke
case passed normal and ReLU numerical checks. Its complete-transfer admission
measurement was **23.715 ms GPU versus 0.026 ms CPU**, so hybrid correctly
rejected this pointwise segment on this UMA driver. The implementation is kept
as a portable batched GPU building block for devices/shapes where it wins; it
does not claim a GPU speedup here and does not change the `gpu_only` contract.

### 2026-08-12: fused GPU Sigmoid gate and refreshed checks

The pointwise Vulkan path now also fuses exact `Conv -> Sigmoid` (shader mode
10). This covers the detector's post-folding pointwise gate form without a CPU
activation pass or an extra activation buffer. Its public low-level API and
hybrid admission key distinguish ReLU, Swish, and Sigmoid; the admission probe
includes complete host upload, dispatch, fence wait, readback, and FP32
comparison with the CPU reference. A shape is selected only when that whole
round trip is no slower than CPU, and `gpu_only` remains unavailable until the
complete detector/recognizer/transformer/CTC graph is device-resident.

On the local AVX-512 / AMD Radeon 780M host, the refreshed Vulkan smoke test
passed the batched, odd-plane (`N=4, C=24, M=40, plane=257`) Conv+Sigmoid
numerical check together with existing binary, affine, Swish, residual, and
shared-memory-tiled pointwise checks. The same run reported:

| Segment admission sample | GPU | CPU | Hybrid selection |
|---|---:|---:|---|
| Batch affine+Swish | 0.15924 ms | 0.29874 ms | GPU |
| Batch `1x1` Conv | 0.16150 ms | 0.02062 ms | CPU |
| Batch `1x1` Conv+Swish | 0.18282 ms | 0.10534 ms | CPU |
| Batch residual `1x1` Conv+Swish | 0.14726 ms | 0.09338 ms | CPU |

These are measured partial-segment results on a UMA GPU, not an end-to-end GPU
claim. A post-change whole-pipeline smoke benchmark on `regression_ppm/en.ppm`
with the tiny model (3 warm-ups, 12 runs) measured CPU **28.584 ms** and
hybrid **27.684 ms** mean. A medium-model check (2 warm-ups, 8 runs) measured
CPU **201.388 ms** and hybrid **206.885 ms**; shared-host variation and partial
coverage mean CPU remains the sensible default for this device. Both modes
preserved OCR output correctness. The low-level AVX-512 kernel regression also
passed normally and with `PPOCR_DISABLE_AVX512=1`, verifying the AVX2 fallback.

Reproduce the verification:

```powershell
cmake --build build --parallel
.\build\ppocr_vulkan_smoke.exe
.\build\ppocr_kernel_smoke.exe
$env:PPOCR_DISABLE_AVX512='1'; .\build\ppocr_kernel_smoke.exe
Remove-Item Env:PPOCR_DISABLE_AVX512
```

### 2026-08-12: diverse-size batch and memory regression

The post-tile regression was widened beyond short English lines. With
recognizer batch size 4 and one recognition worker, the medium model preserved
serial-equivalent OCR across a `16x16` tiny glyph, a `3000x80` wide strip, and
a dense `1920x1080` UI page (124 total results). It measured **5652.199 ms
serial / 5552.878 ms batch** (`1.018x`) and a subsequent two-cycle mixed-size
memory soak used **391,716,864 bytes** after warm-up; cycles ended at
**+6,422,528** and **+184,320 bytes**. This is bounded Windows allocator
retention after the dense page, not a per-shape cache. Small-model CPU and
hybrid checks likewise preserved output parity across `16x16`, `3000x80`,
mixed-language, and dense-page cases. The current GPU graph is still partial,
so page batching is a capacity feature rather than a universal latency win;
the bounded default remains one concurrent page.

The accompanying 1-warmup/7-run `en.ppm` whole-pipeline A/B (same host,
default batch settings) was:

| Model | CPU mean | Hybrid mean | Result |
|---|---:|---:|---|
| tiny | 27.867 ms | 34.334 ms | CPU faster |
| small | 67.981 ms | 79.373 ms | CPU faster |
| medium | 205.998 ms | 214.396 ms | CPU faster |

This outcome is expected for a partial graph with synchronous round-trips;
use `PPOCR_BACKEND=cpu` on this device for current end-to-end throughput. The
six-image mixed-size page-batch regression still produced byte-identical OCR
results versus serial execution and measured **410.192 ms serial**, **373.145
ms batched** at page parallelism 2 (**1.099x**).
For a fuller pipeline A/B on the same shared Windows host (small model,
`ui_dense_1920x1080.ppm`, 119 boxes, 16 CPU workers, recognizer batch 4 /
parallelism 14, one warm-up and three timed runs), CPU measured **1044.384 ms
mean** and hybrid **1081.426 ms mean**. This short noisy sample is recorded as
an end-to-end non-regression check rather than a GPU speedup: the per-shape
admission rule continued to reject GPU segments, and hybrid's small overhead
remains a reason to use `PPOCR_BACKEND=cpu` when the deployment only needs the
current CPU graph.

After the fused Swish GPU path was added, a fresh small-model dense-page A/B
(`ui_dense_1920x1080.ppm`, 119 boxes, batch 4, parallelism 14, 1 warm-up / 3
runs) measured **1170.717 ms hybrid** and **1155.998 ms CPU**. The short run is
intentionally reported as a non-regression observation rather than an
end-to-end GPU speed claim: only eligible BatchNorm-Swish shapes are offloaded,
and scheduling/shape mix can still outweigh an individual accelerated segment.
The per-shape gate continues to preserve correctness and rejects any shape
whose complete transfer/dispatch/readback time loses to CPU.

The CPU fallback for the exact `Sigmoid`, `Swish`, and
`BatchNorm -> Swish` paths is also unrolled into four independent scalar-libm
exponentials. This improves instruction-level parallelism on x86 (including
AVX-512 hosts, where there is still no exact vector `exp` instruction) and
ARM/NEON without adopting an approximate exponential or changing the ONNX
logistic expression. The post-change tiny/small/medium ONNX Runtime text
checks remained **5/5**, **3/3**, and **3/3** respectively.
Fused `BatchNorm -> Swish` now also reuses the source activation when it is at
its final graph use. This removes one recognizer-sized output allocation and
copy on both x86 and ARM/NEON while preserving the affine, exact logistic, and
multiply operation sequence per scalar. It can be A/B tested with
`PPOCR_DISABLE_BN_SWISH_INPLACE=1`. On the shared host, a small-model
three-crop `mixed.ppm` sample was noisy and did not show a reliable latency
win (100.535 ms in-place vs 98.598 ms legacy over 15 runs), whereas medium
measured 278.592 vs 282.793 ms over 10 runs; therefore this is recorded as an
activation-memory/frontier optimization rather than a portable speed claim.

A later experiment to fuse transformer `GEMM -> exact Swish` at row granularity
was A/B tested and intentionally not retained: its benefit was not stable on
small/medium at the end-to-end level. The portable GEMM and exact Swish passes
therefore remain separate, avoiding an unsupported speed claim. This is the
same evidence-first policy used for every hybrid GPU admission.

The same shader now accepts an equal-sized batch in its Vulkan dispatch Y
dimension: independent recognizer-like crops share one command-buffer reset,
submission, completion wait, and H2D/D2H pair instead of paying those costs once per
crop. Each compute invocation handles four contiguous FP32 values (`vec4`) and
keeps only a 1--3 value scalar tail, reducing work-item scheduling overhead on
wide OCR planes. The shader updates the mapped left-hand activation buffer in
place, eliminating the former dedicated output buffer and its activation-sized
copy: this Vulkan primitive now retains two persistent buffers
(activation/result plus compact RHS), rather than three. Persistent mapped
buffers now prefer coherent host-cached memory when the Vulkan driver offers
it (with a portable coherent-memory fallback), grow with 1.5x headroom, and
resize the activation/result independently of the compact RHS. This avoids
needless buffer recreation when recognition width buckets change. Completion uses a submission fence rather
than queue-wide idle, so an individual synchronous mapped readback does not
also wait for unrelated queue work. The batch admission check validates every
item and includes complete batched transfer/dispatch/readback time; hybrid
selects it only when that time is no slower than CPU SIMD. The current Radeon
780M four-crop, 4,099-float, four-stage check measured **0.146 ms GPU versus
0.037 ms CPU** after this batch-path update, so hybrid correctly retained CPU.
This is an implemented
batched Vulkan executor path and building block, not a full GPU OCR claim until
convolution, pooling, resize, normalization, MatMul, and CTC are device-resident.

For repeatable backend A/B measurements, `ppocr_bench` accepts
`PPOCR_BACKEND=cpu|hybrid|gpu` (default: `hybrid`) in addition to
`PPOCR_REC_BATCH_SIZE`, `PPOCR_REC_WIDTH_BUCKET`, and
`PPOCR_REC_PARALLELISM`. Its JSON output records the selected backend. `gpu`
continues to fail explicitly until the full PP-OCRv6 graph is Vulkan-resident;
it never labels a CPU fallback as GPU inference.

The post-optimization Vulkan smoke test on 2026-08-11 passed scalar,
non-workgroup-aligned, 65,537-element, dependent-chain, four-item batch, and
compact NCHW suffix-broadcast batch cases on AMD Radeon 780M Graphics. The
suite now also covers an immutable shared-RHS broadcast batch over two distinct
activations. The broader native regression corpus also
passed **11/11** cases: blank, tiny, wide/tall strips, low-contrast/noisy,
rotated, dense blobs, and 1,920鑴?,080 / 2,560鑴?,440 screenshots. Text parity
against ONNX Runtime passed **5/5** mixed English/Chinese samples. Exact pixel
geometry is intentionally not a strict invariant because equivalent DB contour
rounding can differ by one pixel; recognized text and normal-tolerance
confidence parity are checked.

The CPU convolution path additionally fuses ReLU into the AVX-512/AVX2
2鑴?-valid and 3鑴? convolution stores (both stride-1 and stride-2), which
removes the subsequent full activation read/write pass for detector stems,
the reconstruction head, and the medium detector context head.
`PPOCR_DISABLE_FUSED_CONV_RELU=1` is
available solely for deployment A/B measurement; it retains the former
separate SIMD ReLU pass. On the two-line `en.ppm` workload, a 20-run tiny A/B
was **39.385 ms fused vs 40.599 ms unfused** (about 3.0% lower mean latency).
After the stride-1 extension, a 15/12/12-run A/B measured tiny at **46.692 vs
48.548 ms** (3.8% lower), small at **128.286 vs 136.103 ms** (5.7% lower), and
medium at **312.317 vs 292.898 ms**. The medium result is adverse on this
shared host, so the `PPOCR_DISABLE_FUSED_CONV_RELU` escape hatch remains for
deployment tuning; it is reported rather than presented as a portable gain.
Correctness was rechecked with AVX-512 disabled (AVX2 dispatch), five ORT
text-parity samples, and the 11-case stress corpus.

Adding 2鑴?-valid coverage was remeasured on the same `en.ppm` workload:
15-run tiny was **35.783 vs 37.773 ms** (5.3% lower), medium was **271.887 vs
272.238 ms** (0.1% lower), while small contained a single 738.612-ms scheduling
outlier (median **91.147 vs 94.687 ms**). The median is the useful comparison
for that shared-host run; all raw mean/median/p95 numbers remain available from
`ppocr_bench` JSON for deployment-specific decisions.

The executor now also transfers a dead activation's backing storage through
internal `Reshape`, `Squeeze`, and `Unsqueeze` nodes instead of copying floats
solely to change tensor metadata. It retains the existing last-use and graph
output guards, so aliases never escape into a live consumer or public result.
The terminal-CTC recognizer executor uses the same transfer path, covering the
small/medium transformer reshape boundaries rather than only the generic
executor. ORT text parity passed tiny **5/5** and small/medium **3/3** sampled
English, Chinese, and mixed pages after this change. The expanded 11-image
stress suite also passed. A size-changing memory soak across `en.ppm` and
`ui_dense_1920x1080.ppm` remained bounded: tiny stabilized around 80 MiB
private bytes (at most +3.61 MiB above its post-warmup sample) and medium
stayed below its 308.93 MiB post-warmup sample on the measured cycles; small
remained around 105--112 MiB (+6.34 MiB at most after warmup). Dense-page
native throughput was additionally exercised with 119 small-model crops
(1.422 s mean) and 122 medium-model crops (6.009 s mean), confirming the
optimized recognizer path under varied crop widths and bounded batches.

`PPOCR_PROFILE=1` now additionally emits a separate `ppocr CTC operator
profile` for the recognizer's terminal-CTC executor. This exposes the actual
small/medium transformer costs (rather than only detector `Run()` nodes) while
keeping the release path free of timing/map allocations. On the 123-crop dense
page, tiny-model batch-size tuning measured `rec_batch_size=4` at **323.479
ms**, versus 348.242 ms (8) and 339.535 ms (16); the default remains 4. A
parallelism sweep for that same batch size measured 498.854 ms (4 workers),
392.288 ms (8), 339.808 ms (14), and 329.693 ms (24). The 24-worker result is
only a small host-specific gain with a higher concurrent activation frontier,
so the library retains its more memory-efficient default of 14 workers.

The current transformer kernels add two further low-memory CPU optimizations.
Wide AVX-512/AVX2 GEMM projections now retain four output vectors in registers
across the entire K reduction, eliminating the former output load/store for
every K item; this directly targets the `FusedMatMulBias` vocabulary head.
The repeated LayerNorm `Pow(x,2)` nodes now use a SIMD square kernel and reuse
their dead centered activation in place, avoiding both `powf` and a second
`[N,T,C]` allocation. The latter is exact multiplication, not an approximate
math mode. On the latest 16-worker 20-run paired `en.ppm` measurement, the
current release measured **41.718 ms tiny, 93.439 ms small, and 270.362 ms
medium**, against ONNX Runtime CPUExecutionProvider **144.039, 272.186, and
640.506 ms** respectively. That is **0.290x / 0.343x / 0.422x** C++/ORT
latency (about **71.0% / 65.7% / 57.8% lower** C++ latency) on this shared
host. Exact ORT text parity after these changes passed tiny **5/5** and
small/medium **3/3** English, Chinese, and mixed-image cases.

A post-change changing-size memory soak (`en.ppm` plus the 1,920鑴?,080 dense
page) remained bounded. Tiny's warm baseline was **78,929,920 bytes** and
its largest measured delta was **+1,032,192 bytes**; small's baseline was
**107,143,168 bytes** and its largest delta **+6,766,592 bytes**; medium was
at or below its **284,958,720-byte** warm baseline in both measured cycles.

The five transformer LayerNorm chains in each small/medium recognizer are now
recognized at model load and collapsed from `ReduceMean 閳?Sub 閳?Pow 閳?ReduceMean 閳?Add 閳?Sqrt 閳?Div 閳?Mul 閳?Add` into one affine last-axis kernel.
It creates only the final `[N,T,C]` activation, eliminates centered/squared/
normalized temporary tensors, and keeps the standard `sqrt`, division, gamma,
and beta operation ordering. On the same 16-worker 20-run `en.ppm` protocol,
the resulting C++ measurements were **36.683 ms tiny, 75.827 ms small, and
229.388 ms medium**, while the paired ORT figures remain **144.039, 272.186,
and 640.506 ms**. That is **74.5% / 72.1% / 64.2% lower latency** for the
native path on this host. Post-change text parity passed tiny **5/5** and
small/medium **3/3**. A fresh multi-size soak measured small from a
**109,473,792-byte** warm baseline with a maximum **+4,493,312-byte** delta;
medium began at **276,164,608 bytes** with a maximum **+929,792-byte** delta.

`MatMul + bias + Sigmoid + Mul` gates in the small and medium transformer MLPs
are also fused into a single bias-initialized MatMul followed by in-place Swish;
each model matches this graph pattern twice. It removes the intermediate bias
and sigmoid activations without changing FP32 operation order. Text parity
after that change passed tiny **5/5** and small/medium **3/3** against ONNX
Runtime. A fresh CPU-only 16-worker sanity run (5 warmups / 20 runs, subject
to documented Windows scheduling variation) measured **43.778 ms tiny,
97.592 ms small, and 277.404 ms medium** on `en.ppm`; the established paired
ORT baseline above remains the authoritative cross-engine comparison.

The fused MLP path is now additionally specialized for its actual PP-OCRv6
rank-2 constant projection: it derives the flattened row count directly and
avoids generic batch-shape/stride bookkeeping before invoking GEMM, then
applies Swish in the same destination. Generic broadcasted `MatMul` retains
the portable implementation, but its batch strides are now computed once per
node instead of allocating them once per batch item. The latest validation
covered all three models across **16 PPM inputs**: tiny and small completed all
16 in one run, while medium completed all 16 in two timeout-bounded eight-image
runs. The corpus spans 16x16, 80x3000, 3000x80, noisy/low-contrast, rotated,
dense, blank, Chinese/English/mixed, and 1920x1080 / 2560x1440 screenshot
inputs. Exact ORT text parity was rechecked as tiny **5/5** and medium **3/3**
after the specialisation; small's existing **3/3** exact gate remains passing.

The AVX-512 `1x1 Conv + residual Add` path now has a real eight-output
register tile rather than issuing two four-output kernels. Each 16-float input
vector is loaded once and contributes to eight independent output channels;
the residual is added only after each channel reduction, retaining the ONNX
`Conv` then `Add` operation order. On the same 16-worker, 5-warmup/20-run
medium `en.ppm` A/B, the enabled tile measured **296.941 ms** (median 295.073)
versus **317.217 ms** (median 317.510) with
`PPOCR_DISABLE_AVX512_POINTWISE8=1`: a **6.4% mean** reduction on this host.
The current post-change CPU sanity sweep measured **42.518 ms tiny, 86.810 ms
small, and 296.941 ms medium**; shared-host variance means the established
paired C++/ORT table remains the authoritative engine comparison. Tiny 5/5
and medium 3/3 ORT text parity were repeated after the kernel change; the
medium three-size memory soak (16x16, 3000x80, dense 1920x1080) started at
**330,235,904 bytes** and ended one cycle at **+4,177,920 bytes**. Vulkan's
scalar, odd-size, chain, batch, and compact-broadcast smoke cases also passed.

Recognizer attention transpose now has exact rank-4 specialisations for the
two PP-OCRv6 permutations `[0,2,1,3]` and `[0,1,3,2]`. The former moves its
contiguous final feature vector with `memcpy`; both avoid the generic
per-float four-axis divide/modulo index decoder. After the change, ORT text
parity passed tiny **5/5**, small **3/3**, and medium **3/3**. On the current
16-worker 5-warmup/20-run CPU samples, tiny/small/medium measured **42.771 /
87.300 / 261.648 ms** on `en.ppm`. The current Vulkan smoke run also passed;
the Radeon 780M admission samples remained GPU-slower end to end and hybrid
therefore retained CPU SIMD rather than claiming an unsupported speedup.

Attention Q/K/V splitting now also has a safe contiguous `Slice` fast path:
when ONNX Slice selects an axis-0 unit-stride range while preserving every
remaining axis, it transfers the complete `[axis-0 row]` span with one
`memcpy` rather than doing per-value stride division/modulo. This exactly
matches the stacked Q/K/V slices used by the small and medium recognizers;
all non-contiguous or non-unit-step layouts retain the general ONNX Slice
implementation. The post-change exact ORT gates passed tiny **5/5**, small
**3/3**, and medium **3/3**. A wider eight-case small-model corpus was also
run; six cases passed (blank, English, Chinese, mixed, line, and noisy), while
the deliberately difficult dense-screen and rotated-text samples are currently
recorded as known OCR-output divergences from ORT rather than excluded or
misreported as a passing parity gate.

The recognizer's remaining final-head layout exchange `[N,T,C] -> [N,C,T]`
now also uses a direct three-axis loop instead of generic shape/stride index
decoding. Together with the rank-4 attention paths, this covers every fixed
transpose permutation used by the shipped tiny/small/medium recognizers while
retaining the generic implementation for other ONNX graphs. The post-change
parity gates again passed tiny **5/5**, small **3/3**, and medium **3/3**.
The medium short benchmark remained shared-host variable (a fresh 20-run
sample measured 302.714 ms), so no additional end-to-end speed claim is made
from that isolated run; the change is retained for lower overhead on the
repeated transpose-heavy recognizer paths.

The latest backend A/B sample on the two-line tiny workload (`en.ppm`, 3
warmups / 15 runs) measured **41.533 ms hybrid** versus **42.553 ms CPU**.
This small 2.4% difference is within normal shared-host variability: the
Radeon admission measurements above declined all tested segments, so it must
not be interpreted as a standalone GPU OCR speedup. On the 123-crop dense
page (2 warmups / 7 runs), hybrid measured **311.501 ms** and CPU **314.929
ms**; both paths preserve exact CPU inference for the parts of PP-OCRv6 not
yet resident on Vulkan.

After the LayerNorm/transformer update, the denser 123-crop 1,920鑴?,080 workload
was repeated for 3 warmups / 10 runs: **299.313 ms hybrid** (median 297.708)
and **293.864 ms CPU** (median 292.125). The Radeon 780M's independently
measured admissions still rejected the individual Vulkan elementwise segments
(for example the compact broadcast batch was 0.106 ms GPU versus 0.010 ms
CPU), so this close whole-pipeline comparison is recorded as shared-host
variance rather than claimed GPU acceleration.

After the Vulkan batch-buffer/fence update, the same 123-crop dense page was
rechecked for 3 warmups / 10 runs: **346.120 ms hybrid** (median 342.037) and
**347.319 ms CPU** (median 345.431). This near tie is shared-host variance,
not GPU acceleration: all current per-shape admissions still rejected the
Radeon 780M elementwise segments. The smoke suite passed scalar, odd-length,
fused-chain, compact broadcast, and four-crop batch cases; its current
end-to-end admission samples were 65,537-float binary **0.256 vs 0.028 ms**,
four-stage chain **0.169 vs 0.032 ms**, four-crop chain **0.146 vs 0.037 ms**,
and compact broadcast **0.150 vs 0.016 ms** (GPU vs CPU). `gpu_only` was also
rechecked and continues to fail explicitly because the full Vulkan OCR graph
has not yet been implemented.

After the in-place mapped-buffer update, Vulkan smoke was rerun on 2026-08-11
and passed scalar, odd-length, fused-chain, four-crop batch, and compact
broadcast batch outputs on AMD Radeon 780M Graphics. Current full-transfer
admission samples remained transfer-bound: binary **0.333 vs 0.034 ms**,
four-stage chain **0.250 vs 0.047 ms**, four-crop chain **0.133 vs 0.031 ms**,
and compact broadcast **0.161 vs 0.015 ms** (GPU vs CPU). Accordingly hybrid
still executes those isolated segments with CPU SIMD. The saved mapped-buffer
memory is real, but it is not represented as a whole-OCR GPU speedup.

The fused pointwise-convolution channel-bias path now uses the existing
runtime-dispatched scalar-broadcast kernel per NCHW plane. This removes its
scalar inner loop while preserving the exact `Conv` then `Add` order, with
AVX-512, AVX2, NEON, and portable scalar coverage selected dynamically. The
follow-up exact ORT gates passed tiny **5/5**, small **3/3**, and medium
**3/3**. A post-change 16-worker 5-warmup/20-run CPU sanity sample measured
**41.587 / 84.744 / 259.606 ms** for tiny/small/medium on `en.ppm`; Windows
scheduling varies enough that the paired C++/ORT table above remains the
authoritative cross-engine claim. A small dynamic-size memory soak across
16鑴?6, 3000鑴?0, and dense 1920鑴?080 inputs started at **110,268,416 bytes**;
two cycles ended at **-2,564,096** and **+2,109,440 bytes** from that baseline,
respectively, with no monotonic retained-shape growth.

The latest exact-mode CTC confidence scan now issues four independent
scalar-libm exponentials before adding them in the original lane order. This
preserves the 6,906-class Softmax confidence calculation and CTC output while
allowing x86 (including AVX-512-capable systems) and ARM cores to overlap exp
latency. Build plus ONNX Runtime text gates passed tiny **5/5**, small **3/3**
and medium **3/3** after the change. A fresh 16-worker, 5-warmup/20-run CPU
sanity pass on `en.ppm` measured **36.458 / 70.745 / 218.498 ms** for
tiny/small/medium. The same revision's diverse-size CPU sweep recorded:

| Model | `en.ppm` (2 boxes) | `wide_strip_3000x80.ppm` | `ui_dense_1920x1080.ppm` |
|---|---:|---:|---:|
| tiny | 33.555 ms | 23.269 ms | 278.795 ms (123 boxes) |
| small | 74.727 ms | 57.801 ms | 1136.849 ms (119 boxes) |
| medium | 221.817 ms | 150.859 ms | 4626.259 ms (122 boxes) |

Those short varied-size samples are intentionally recorded as local CPU
capacity data, not substituted for the repeated paired C++/ORT table. The
post-change small mixed-size two-cycle memory soak (16x16, 3000x80, and
1920x1080 dense UI) measured **-12,288 bytes** then **+5,222,400 bytes** from
a 109,170,688-byte baseline, showing no monotonically growing allocation per
input shape. Vulkan smoke also passed on AMD Radeon 780M Graphics; full
transfer admission remained CPU-faster (binary **0.13716 vs 0.01282 ms**,
four-stage batch **0.10178 vs 0.02728 ms**, GPU vs CPU), so hybrid correctly
continued to avoid increasing latency or CPU work with an unprofitable GPU
offload.

For AVX-512 x86 systems, the detector's hot stride-2 3x3 convolution now
forms its sixteen every-other-pixel samples from two contiguous row loads and
an AVX-512 lane permute, rather than issuing a gather for each 3x3 tap. The
packed-load implementation remains dynamically switchable with
`PPOCR_DISABLE_AVX512_STRIDE2_PACKED=1`, preserving the prior gather kernel
for any deployment where its microarchitecture wins. On this host's medium
`en.ppm` 16-worker 5-warmup/20-run A/B, packed loads measured **219.768 ms**
(median 216.588) versus **274.781 ms** (median 229.537) for gathers. The mean
is scheduler-sensitive, but both median and mean favor the packed path. Its
full exact ORT text gates passed tiny **5/5**, small **3/3**, medium **3/3**;
a medium mixed-size memory cycle (16x16, 3000x80, dense 1920x1080) ended
**98,304 bytes below** a 277,999,616-byte baseline. The Vulkan smoke suite
also remained passing; its current full-transfer admissions still correctly
select CPU for this Radeon 780M, so this CPU optimization never changes the
GPU-only or hybrid correctness contract.

An AVX-512 eight-output-channel 3x3 stride-one detector kernel shares every
interior input vector across eight output reductions while retaining scalar
border handling and the exact per-channel reduction order. A refreshed
24-run, 2-warmup CPU A/B on the current host shows the current restricted
default predicate is beneficial for all three `en.ppm` models: tiny
**32.010 -> 31.166 ms** (1.027x), small **66.480 -> 65.099 ms** (1.021x),
and medium **207.772 -> 203.876 ms** (1.019x), four-output -> tile-8.
`PPOCR_DISABLE_AVX512_CONV3X3_TILE8=1` remains the deployment fallback for
machines whose AVX-512 frequency characteristics differ. The stricter shape
gate still routes smaller layers, odd channel tails, AVX2, NEON, and scalar
targets to their existing proven paths.

The adjacent AVX-512 eight-output `1x1` projection tile was also refreshed as
a negative-control A/B, rather than broadening its predicate speculatively.
With the current `192x192` gate, four-output -> eight-output measured tiny
**32.097 -> 34.343 ms** (regression), small **70.746 -> 66.848 ms** (win),
and medium **216.021 -> 214.452 ms** (small win), using the same 2-warmup,
24-run CPU procedure. Consequently the shape-local `192x192` default and
`PPOCR_DISABLE_AVX512_POINTWISE8=1` fallback remain unchanged: a single global
threshold cannot improve every tier on this client CPU. This is deliberately
recorded rather than claiming a universal pointwise gain.

The detector FPN's common nearest-neighbour 2x resize now writes four source
pixels as eight direct replicated stores before vertically copying the row.
This keeps the exact NCHW output layout while removing per-pixel
`std::fill_n` setup in every 2x FPN plane; uncommon integer resize factors
retain the existing general replication path. A fresh default medium
16-worker 5-warmup/20-run `en.ppm` sample measured **209.972 ms** (median
210.745 ms). Exact ORT text regression again passed tiny **5/5**, small
**3/3**, medium **3/3**, and Vulkan smoke stayed passing. A two-cycle tiny
mixed-size soak across 16x16, 3000x80 and 2560x1440 inputs stayed bounded at
**+1,150,976** then **+2,654,208 bytes** relative to its 82,276,352-byte
post-warmup baseline; this represents Windows heap retention, not a
per-inference activation leak.

Recognition batch construction no longer allocates and copies a temporary
`vector<CropBounds>` for every batch. Same-width buckets now retain contiguous
crop bounds and result indices once, and `RecInputBatch` consumes a direct
span. This removes hundreds of small allocations/copies on dense pages while
preserving crop order, width bucketing, bounded batch size, and concurrent
recognizer behavior. The medium 122-box dense 1920x1080 smoke workload
completed correctly after this change (a short 2-warmup/5-run local sample
was **5541.268 ms**, subject to the documented scheduler/cache variability).
Exact ORT text parity remained tiny **5/5**, small **3/3**, medium **3/3**;
Vulkan scalar/chain/batch/broadcast smoke passed. A medium mixed-size
16x16/3000x80/1920x1080 memory cycle completed without a failure; its
post-warmup private-byte result is retained as capacity data rather than
claimed as a leak-free steady-state proof because the Windows allocator kept
10,149,888 bytes after this deliberately large dense workload.

The Vulkan binary runtime now recognizes immutable ONNX initializer operands.
When hybrid dispatches an eligible elementwise node, its RHS remains in the
persistent mapped Vulkan buffer across later activations of the same model,
avoiding a repeated host copy while retaining a separate activation/result
buffer. Dynamic RHS tensors always copy, and the existing full end-to-end
admission policy is unchanged: a cached constant does not qualify GPU unless
the whole H2D/dispatch/D2H segment is no slower than CPU. The smoke suite now
checks two distinct activations against a cached 65,537-float immutable RHS;
it passed on AMD Radeon 780M Graphics along with scalar, chain, batch and
broadcast cases. Current admissions remain CPU-faster on this UMA host, so
hybrid correctly preserves CPU SIMD rather than claiming a GPU OCR speedup.
The default CPU executor's tiny **5/5**, small **3/3**, and medium **3/3**
ORT text gates were rerun after the backend API change.

### Batched Vulkan residual pointwise block

The Vulkan backend now has a verified batched `1x1 Conv + residual Add`
primitive, with a `Conv + Add + ReLU` variant. It maps equal-width recognition
crops to Vulkan's Y dimension, keeps learned `[M,C]` weights and bias cached
when they are ONNX initializers, and uploads the residual into a separate
read-only buffer so the shader never aliases an input with its output. This
is also wired into the existing fused CPU graph nodes in `hybrid` mode, while
the shape-specific admission test includes both live activation uploads,
parameter transfer, dispatch, readback, and an FP32 CPU comparison.

On 2026-08-11, the AMD Radeon 780M smoke test validated normal and ReLU
residual pointwise results for four NCHW batches, including a 257-element
non-vector-aligned spatial plane. Its full-transfer admission measurement for
the `N=4, C=24, M=40, plane=257` `Conv+Add+ReLU` case was **19.1844 ms GPU**
versus **0.0326 ms CPU**, so hybrid correctly rejected the GPU segment. The
normal pointwise case was likewise **19.1564 ms GPU** versus **0.03072 ms
CPU**. These figures are deliberately recorded as a negative end-to-end
result: the new GPU kernel is numerically covered and batch-capable, but
standalone host-visible uploads/readback make it inappropriate on this UMA
device. `gpu_only` remains explicitly unavailable until the complete OCR
graph can execute device-resident without CPU fallbacks.

After this change, the tiny five-image C++/ONNX Runtime text regression again
passed **5/5**. The six-image mixed-size page-batch smoke (151 detected text
regions, page parallelism two) completed in **381.767 ms** versus **408.840
ms** serial, a **1.071x** scheduler speedup on the local shared host.

The residual-ReLU AVX-512 eight-output entry point was also exercised on the
same host. Its real eight-live-accumulator version passed the five-image text
gate, but did not demonstrate a stable end-to-end win: `mixed.ppm` small was
**86.242 ms** versus **82.700 ms** with the default four-output tile, and the
medium results varied between runs. It therefore continues to delegate to the
proven four-output residual-ReLU tile instead of becoming an unvalidated
default. This keeps AVX-512 dynamic and safe while avoiding a register-pressure
regression on both tiny and small workloads.

The current medium mixed-size memory soak (`16x16`, `3000x80`, `80x3000`, and
the 151-region `1920x1080` UI page; two cycles) established a post-warm-up
private-byte baseline of **347,013,120 bytes**. The next cycles were
**+303,104 bytes** and **+3,276,800 bytes**. This bounded allocator-capacity
movement is recorded as a Windows high-water observation rather than a general
leak proof; importantly it did not grow once per dynamic shape or per crop
batch.

### 2026-08-11: GPU residual-vector and non-ReLU batch fusion follow-up

The residual pointwise shader now vectorizes its common four-contiguous-spatial
lane case as well as the non-residual pointwise shader.  It performs
`1x1 Conv + residual Add (+ ReLU)` in registers and writes one final NCHW
result, retaining a scalar tail only at plane boundaries and odd widths.  The
graph loader also recognizes the strict dynamic `1x1 Conv -> equal-shape Add`
form without a following ReLU as `FusedPointwiseConvAdd`.  This matters for
the medium recognizer: a CPU profile of `mixed.ppm` observed fused non-ReLU
shortcut nodes such as `Conv.9_add`, `Conv.20_add`, `Conv.50_add`, and
`Conv.31_add`, so they now avoid a materialized Conv result plus a separate
Add pass on both CPU and eligible hybrid paths.  Initializer/broadcast Adds
remain excluded, preserving ONNX bias semantics.

The rebuilt Vulkan smoke suite passed all binary, broadcast, affine, normal
pointwise, residual pointwise, and residual-ReLU numerical checks on AMD
Radeon 780M Graphics.  Its latest complete-transfer measurements were
**21.0106 ms GPU vs 0.02642 ms CPU** for `N=4,C=24,M=40,plane=257` pointwise
Conv and **14.2844 ms GPU vs 0.03048 ms CPU** for residual+ReLU; hybrid
therefore rejected both exactly as designed.  The separate tiny five-image
C++/ORT text gate passed **5/5**, and the six-page mixed-size batch smoke
produced identical nested results in **81.021 ms** versus **122.164 ms**
serial (**1.508x**).  A short `mixed.ppm` end-to-end check measured **34.379
ms CPU** versus **40.314 ms hybrid** (one warm-up, three runs), so current
deployment guidance remains CPU for this UMA GPU; these GPU additions are
correct, batched building blocks rather than an unsupported full-GPU claim.

### 2026-08-11: Vulkan mapped-memory policy revalidation

The persistent Vulkan buffers now explicitly score host-coherent memory types
for this synchronous hybrid runtime: `HOST_CACHED` coherent memory is selected
first, then host-visible device-local coherent memory, then a portable coherent
fallback. This is not merely a theoretical ordering. An A/B on the Radeon 780M
showed that preferring host-visible `DEVICE_LOCAL` for every mapped buffer made
the complete binary segment about **10.65 ms GPU vs 0.0195 ms CPU**, while the
cached mapped policy restored it to **0.137 ms GPU vs 0.015 ms CPU**. The
device-local mapping was therefore rejected; shader-side device locality does
not compensate for slower CPU copies in an H2D/dispatch/D2H hybrid segment.

With the restored policy, the refreshed smoke suite passed every numerical
case. Its four-crop affine-Swish segment was **0.13832 ms GPU vs 0.29722 ms
CPU**, so hybrid selects Vulkan for that shape; the pointwise and residual
pointwise probes remained transfer-bound (**20.6638 vs 0.0269 ms**, and
**14.3158 vs 0.03158 ms**, respectively) and are correctly retained on CPU.
This is the required behavior for `hybrid`: prefer GPU only when the measured
full segment is no slower, while avoiding a silent fallback in `gpu_only`.

Post-change correctness evidence includes small and medium C++/ONNX Runtime
text parity on `en.ppm` (**1/1** each) and a medium mixed-size memory soak over
`16x16`, `mixed`, and the `1920x1080` 151-region UI image. Its post-warm-up
private-byte baseline was **361,832,448** and the next cycle was
**-7,499,776 bytes** from that baseline, consistent with a bounded allocator
high-water rather than retained dynamic workspaces. A six-page tiny batch
regression still returned byte-identical nested results; the shared-host run
measured **126.196 ms serial / 108.076 ms batch (1.168x)**. Timing varies with
contention, so this documents correctness and a measured throughput win rather
than a portable promise.

### 2026-08-11: batched GPU Conv+Swish fusion

### 2026-08-12: tiled batched Vulkan pointwise convolution

The batched Vulkan `1x1` convolution shader now has a workgroup-tiled path
for `C <= 1024`. One 256-thread workgroup owns four output channels and a
256-pixel spatial tile, cooperatively stages the four immutable `[C]` weight
rows in shared memory, then produces four contiguous FP32 values per lane.
This removes the former per-`vec4` weight streaming in the common NCHW
pointwise case. The same path covers Conv, Conv+ReLU, Conv+Swish, and the
equal-shaped residual variants; odd plane tails still use an exact scalar
write and wider channel counts retain the established generic shader path.

This is a real GPU-kernel and batch-throughput improvement, not an invented
end-to-end GPU win. On the local AMD Radeon 780M, the verified four-crop
`N=4,C=24,M=40,plane=257` smoke measurement improved from the earlier
multi-millisecond pointwise dispatches to **0.16412 ms GPU** for Conv and
**0.15202 ms GPU** for Conv+Add+Swish. CPU SIMD still measured **0.02170 ms**
and **0.09334 ms**, respectively, so the complete H2D/dispatch/D2H admission
rule correctly retains CPU on this UMA host. The smoke suite validates all
normal, ReLU, Swish, residual, and non-vector-aligned outputs before timing.

The rebuilt tiny four-page scheduler check (`en`, `line_en`, `mixed`, `zh`;
batch size 4, recognizer parallelism 1, page parallelism 2) remained output
identical after the shader change. This run measured **128.114 ms serial /
88.105 ms batch (1.454x)** on CPU and **125.542 ms / 77.657 ms (1.617x)**
in hybrid mode. These short shared-host samples are scheduler evidence, not a
replacement for the paired ONNX Runtime table above. `gpu_only` remains
explicitly unavailable until the complete PP-OCRv6 graph is device-resident.

For a post-change whole-pipeline sanity A/B on `en.ppm` (16 CPU workers,
batch size 4, recognizer parallelism 1, three warm-ups and ten timed runs),
hybrid measured **34.387 / 72.977 / 216.534 ms** for tiny/small/medium.
The separately sampled CPU tiny run was **36.125 ms**; due shared-host noise
and partial-graph selection these are not substituted for the longer paired
CPU-versus-ORT results. The relevant GPU proof is the verified tiled primitive
and its conservative per-shape full-transfer admission, not an unsupported
claim that all three models now execute fully on the Radeon GPU.

The Vulkan pointwise kernel now also supports the graph-fused PP-OCR pattern
`1x1 Conv -> Sigmoid -> Mul` as an exact FP32 `1x1 Conv + Swish` operation.
Four equal-width NCHW crops share one submission; learned `[M,C]` weights and
`[M]` bias remain compact and can be retained in the persistent parameter
buffers. This removes the CPU-side full-output Swish traversal when a shape is
admitted to Vulkan, without changing the portable AVX-512/AVX2/NEON fallback.

The smoke test verifies the fused output at every element for a four-crop,
non-vector-aligned `N=4, C=24, M=40, plane=257` workload. On the local Radeon
780M its full synchronous H2D/dispatch/D2H measurement was **12.2472 ms GPU
versus 0.13918 ms CPU**, so `hybrid` correctly retains CPU SIMD for that shape.
This is nevertheless a material batch-capable GPU graph building block: a
device/shape is allowed to select it only after this complete measurement and
numerical validation; it is not represented as an end-to-end GPU OCR win on
the present UMA system. The expanded tiny C++/ONNX Runtime text gate passed
**5/5** (`en`, English line, Chinese line, mixed, Chinese page), and the
six-page batch regression preserved identical OCR results at **123.546 ms
serial / 86.991 ms batch (1.420x)** on this run.

### 2026-08-11: shortcut Swish fusion and broader regression pass

Small and medium recognizers contain an inverted-residual suffix
`BatchNorm -> Swish -> Add(shortcut)`. The loader now recognizes this after
the first fusion sweep and emits one `FusedBatchNormSwishAdd` node. Its CPU
kernel produces the affine, exact logistic Swish and equal-shaped residual Add
in one traversal, reusing the dying BatchNorm input allocation. This removes
the intermediate Swish feature map and the separate Add pass. It is protected
by `PPOCR_DISABLE_BN_SWISH_ADD_FUSION=1` for A/B work; a short medium
`mixed.ppm` CPU check (one warm-up / three runs) measured **242.378 ms** fused
versus **260.514 ms** unfused (**7.0% lower latency**), with expected shared
host variance.

Vulkan now exposes the matching batched `channel-affine + Swish + Add`
primitive. Its smoke test validates every output of four `8192`-float NCHW
crops. On the Radeon 780M, complete transfer timing was **0.14988 ms GPU vs
0.14072 ms CPU**, so hybrid correctly chose CPU for that exact case. The graph
path has the same cached no-slower admission rule: it may use Vulkan only when
the full H2D/dispatch/D2H measurement and FP32 validation win; otherwise it
uses the CPU fused path and does not add GPU overhead.

Current post-change evidence: tiny C++/ORT text parity remained **5/5** over
English, Chinese and mixed samples; small and medium each passed an independent
ORT text check on `en.ppm` (**1/1** each). A six-page varied-content tiny batch
regression remained result-identical at **126.926 ms serial / 96.686 ms batch
(1.313x)**. A medium memory soak spanning `16x16`, mixed text, and a dense
`1920x1080` page finished one cycle **7,614,464 bytes below** its
376,975,360-byte post-warm-up baseline, consistent with bounded workspace reuse.

### 2026-08-12: batched Vulkan HardSwish coverage

The graph-fused detector pattern `1x1 Conv -> HardSigmoid -> Mul` now reaches
the Vulkan pointwise kernel as one exact `Conv+HardSwish` dispatch. The shader
receives the ONNX `alpha` and `beta` as raw FP32 push constants and writes
`x * clamp(alpha*x + beta, 0, 1)` directly to the final NCHW output. It uses
the same four-crop Y-dimension batch mapping, shared-weight tile, odd-plane
tail handling, persistent parameter cache, and complete H2D/dispatch/D2H
admission policy as the other pointwise variants. CPU retains its
AVX-512/AVX2/NEON/scalar exact fallback.

The post-change Radeon 780M smoke suite validated every value of an odd-plane
`N=4, C=24, M=40, plane=257` batch. The full synchronous admission sample was
**0.15912 ms GPU / 0.02794 ms CPU**, so this UMA host correctly keeps that
segment on CPU; no end-to-end GPU OCR speedup is claimed. A seven-image tiny
CPU-versus-hybrid output gate (`en`, mixed, CJK, tiny, wide, low-contrast,
blank) passed **7/7** with identical decoded output. With recognizer batch size
4, recognizer parallelism 1, and page parallelism 2, the seven-image hybrid
batch gate preserved output identity and measured **337.565 ms serial /
234.153 ms page-batch (1.442x)**. This is scheduler throughput evidence on
the local machine, not a general GPU performance promise.

### 2026-08-12: three-size CPU/hybrid and memory revalidation

The current build was retested after the Vulkan HardSwish integration with
the official tiny, small, and medium model pairs. The seven-image C++/ONNX
Runtime text gate passed **7/7 for every model** (English, mixed Latin,
CJK, `16x16`, `3000x80`, low-contrast, and blank). The broader tiny stress
corpus also passed **11/11**, including rotated text, dense blobs, and
`1920x1080`/`2560x1440` screenshots.

On `en.ppm` (2 warm-ups, 10 timed runs, 16 CPU workers, recognizer batch 4,
one recognizer worker), the local CPU/hybrid recheck was:

| Model | CPU mean | Hybrid mean | CPU / Hybrid decoded output |
|---|---:|---:|---|
| tiny | 35.164 ms | 34.684 ms | identical |
| small | 71.442 ms | 71.286 ms | identical |
| medium | 205.222 ms | 225.598 ms | identical |

Hybrid is therefore not described as a universal win: each Vulkan segment is
still admitted independently only when its verified complete transfer,
dispatch, and readback time is no slower than the CPU SIMD equivalent. The
medium mixed-size memory soak (`16x16`, mixed text, dense `1920x1080`; two
cycles) used 367,849,472 private bytes after warm-up and ended at
+4,378,624 bytes, consistent with bounded allocator high-water behavior.

### 2026-08-11: batched residual Conv+Swish GPU fusion

The Vulkan batch kernel now also supports the full graph-proven recognizer
suffix `1x1 Conv -> Add(shortcut) -> Sigmoid -> Mul` as mode 9:
`1x1 Conv + residual Add + exact Swish`. Equal-width OCR crops share its Y
dispatch dimension; the learned `[M,C]` weights and `[M]` bias remain compact
and cacheable, and the shortcut is read once before the final device write.
The CPU fallback adds `PointwiseConvAddSwish`, keeping the Conv+shortcut in
the existing AVX-512/AVX2/NEON fused path and overwriting that single output
allocation with exact Swish. Set
`PPOCR_DISABLE_POINTWISE_ADD_SWISH_FUSION=1` to restore the unfused ONNX
sequence for an A/B comparison.

The extended Vulkan smoke test validates every value of an odd-plane
four-crop `N=4,C=24,M=40,plane=257` batch. On the local Radeon 780M, its full
H2D/dispatch/D2H comparison measured **12.3933 ms GPU vs 0.09822 ms CPU**,
so hybrid correctly rejected it. The currently profiled small PP-OCRv6 export
does not emit this exact four-node suffix, so no end-to-end OCR speed claim is
made for the new matcher yet; it is retained as a verified batch-capable
building block for compatible PP-OCRv6 exports, not an unsupported GPU speed
claim on the present UMA host.

### Tiny / small / medium comparison

The same complete two-line OCR pipeline was also measured with the official
three PP-OCRv6 model sizes on this 16-thread AVX-512-capable host.  Model load
is excluded from steady-state latency. The native and ONNX Runtime commands
were run separately with the same PPM input and matching dictionaries. The
following exact-Erf run is the current repeated baseline; `PPOCR_APPROX_GELU`
was unset for every native measurement.

| Model | Native runs / warmup | Pure C++ mean | ORT mean | Native / ORT | Native FPS | ORT FPS |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 20 / 5 | 33.322 ms | 120.106 ms | 0.277x | 30.010 | 8.326 |
| small | 20 / 5 | 83.639 ms | 184.190 ms | 0.454x | 11.956 | 5.429 |
| medium | 20 / 5 | 281.236 ms | 428.584 ms | 0.656x | 3.556 | 2.333 |

This current default-exact run is a same-host comparison with matching model
files and image. It shows lower C++ latency for tiny, small, and medium
(72.3%, 54.6%, and 34.4% respectively). The commands were run as paired
engine measurements, but Windows scheduling still affects individual samples;
the complete JSON output should be retained when refreshing capacity numbers.

The earlier approximate-GELU deployment experiment is retained below as a
separate, opt-in result rather than being conflated with default ONNX-equivalent
execution:

| Model | Native runs / warmup | Pure C++ mean | ORT mean | Native / ORT | Native FPS | ORT FPS |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 10 / 3 | 34.125 ms | 137.535 ms | 0.248x | 29.304 | 7.271 |
| small | 8 / 3 | 95.049 ms | 226.928 ms | 0.419x | 10.521 | 4.407 |
| medium | 8 / 3 | 257.848 ms | 457.041 ms | 0.564x | 3.878 | 2.188 |

All values are from the same 2026-08-11 host and PPM input. The native run
used the persistent 16-worker executor; ONNX Runtime used
`CPUExecutionProvider` with the matching 16-thread budget. The refreshed
measurements show lower mean latency for all three model sizes on this host.
The table below uses the validated optional approximate-GELU deployment setting;
the default exact-Erf paired comparison is documented above.
The opt-in approximation passed the strict 7-image text corpus and the 11-image
size/diversity stress corpus; retain the exact default for applications that
require strict numerical ONNX equivalence.
The specialized stride-2 3x3 gather path remains material for medium:
disabling it measured 2646.179 ms versus 1923.504 ms when enabled with one
requested worker. These are local measurements, not a portable performance
guarantee.

All three model pairs produce the expected two recognized lines in the native
smoke test. The strict 7-image text-parity and 11-image size/diversity stress
gates below use tiny because their known-text expectations correspond to its
dictionary; they exercise input dimensions from 16x16 to 2560x1440.
The latest default-exact verification also forced the dynamic AVX2 fallback
with `PPOCR_DISABLE_AVX512=1`: tiny/small/medium completed at 43.170/113.518/
408.928 ms in a short 8-run sweep. This confirms that AVX-512 remains a
runtime-selected acceleration, not a binary deployment requirement; ARM/NEON
and scalar paths retain the same portable graph implementation.

### Batched constant-weight MatMul fusion

Recognizer attention projections commonly have a batched activation and one
rank-2 immutable weight tensor. The executor now flattens those independent
batch dimensions into one tall GEMM call, avoiding repeated per-batch GEMM
dispatch and allowing the SIMD/NEON GEMM to use its normal row scheduling over
the whole recognition batch. No weights are repacked or duplicated, and every
output row keeps its original increasing-`K` FP32 accumulation order. This is
enabled by default on x86/AVX-512, x86/AVX2, ARM/NEON, and scalar builds;
`PPOCR_DISABLE_MATMUL_BATCH_FOLD=1` provides an A/B fallback.

On the local 16-worker AVX-512 host, exact-GELU `mixed.ppm` measurements with
three recognized crops, batch size four, recognizer parallelism one, three
warmups and fifteen timed runs showed the direct batch benefit for small:
**137.925 ms mean** with the fold versus **195.572 ms** with the fallback
(29.5% lower latency). A separate default scheduler check (batch four,
parallelism fourteen, 3 warmups / 12 runs) measured **41.663 ms tiny**,
**93.915 ms small**, and **340.519 ms medium** on the same input. These
samples are host-specific and the medium workload remains dominated by
convolution, but they confirm that all three model sizes retain the same
correct batched path. Exact ONNX Runtime text parity remained tiny **5/5**,
small **3/3**, and medium **3/3** after the change.
The same fallback path now fuses equal-shape `1x1 Conv -> Add` residual
projections with an AVX2 four-output register tile rather than materializing
the Conv result before a separate Add. A fresh forced-AVX2 3-warmup/10-run
sanity sweep measured tiny/small/medium at 42.446/122.677/434.793 ms on the
shared host; these are coverage and stability samples rather than a paired
ORT comparison. The strict seven-image text gate passed 7/7 with AVX-512
disabled. ARM/NEON has matching four-output fused residual code; it is selected
only when the target compiler exposes NEON, otherwise scalar remains valid.
For AVX-512, large 1x1 projections now use an eight-output register tile
(with four-output/tail handling retained for smaller shapes). It reuses each
loaded input vector across eight output channels and preserves each output
channel's reduction order. The exact-GELU text gate passed 7/7 and the
11-image size/diversity gate passed after this change; the default exact
three-model table above is the associated repeated benchmark.
A 30-run A/B on the same 16-worker host selected this default for the larger
models: small measured 106.047 ms with the eight-output tile versus 109.533 ms
with `PPOCR_DISABLE_AVX512_POINTWISE8=1`; medium measured 301.179 ms versus
313.123 ms. Individual Windows samples remain scheduling-sensitive, but both
the median and mean favoured the eight-output implementation in that repeated
comparison.
The same eight-output technique is implemented for the already fused
`1x1 Conv+ReLU` path, including its exact scalar tail and runtime ISA guards.
It passes the strict text corpus when enabled, but repeated small/medium
end-to-end A/B samples were mixed (and the 30-run medium sample favoured the
default four-output path: 257.914 ms versus 268.169 ms). It is therefore kept
behind `PPOCR_ENABLE_AVX512_POINTWISE_RELU8=1`, rather than silently trading
stable default latency for a workload-specific result.

### Post-fusion comparison refresh

The latest post-fusion same-host refresh (exact Erf-GELU, 16 requested workers,
5 warmups and 20 measured runs per engine) is recorded below. It includes the
new equal-shape `1x1 Conv -> Add` fusion; operator profiling confirmed fused
residual projections in the medium graph (for example `Conv.9_add`,
`Conv.12_add`, `Conv.20_add`, `Conv.50_add`, and `Conv.53_add`). As with the
earlier table, these are sequential paired measurements on a shared Windows
machine, so they document the current build rather than a portable guarantee.

| Model | Pure C++ mean | ORT mean | C++ / ORT | Pure C++ FPS | ORT FPS |
|---|---:|---:|---:|---:|---:|
| tiny | 43.072 ms | 182.025 ms | 0.237x | 23.217 | 5.494 |
| small | 109.535 ms | 244.420 ms | 0.448x | 9.130 | 4.091 |
| medium | 359.697 ms | 642.622 ms | 0.560x | 2.780 | 1.556 |

This refresh measured native latency lower by 76.3%, 55.2%, and 44.0% for
tiny, small, and medium respectively. The observed C++ ranges were
40.202--48.375 ms, 99.140--127.691 ms, and 284.395--396.259 ms; ORT also
showed large host-scheduling variance, especially medium (427.880--1285.338
ms). Keep the original repeated table above for historical comparison and
rerun the paired procedure on the deployment CPU before capacity planning.

The subsequent gate-fusion validation found 13/13, 13/13, and 5/5 fused
HardSigmoid gates in tiny, small, and medium profile runs respectively. Default
and forced-AVX2 strict text parity both passed 7/7, while the 11-image stress
set still passed 11/11. A short shared-host 3-warmup/10-run sanity sweep was
43.159 ms (tiny), 100.075 ms (small), and 416.331 ms (medium); the wide medium
p95 range reflects host scheduling, so these values are deliberately not
presented as a new ONNX Runtime comparison.

The executor now additionally treats a last-use standalone `HardSigmoid` as
an in-place operation. This applies the same exact AVX-512/AVX2/NEON SIMD gate
without allocating a second activation tensor; it is especially relevant to
dynamic image sizes where keeping the graph frontier small lowers transient
memory. The current strict text and 11-image stress checks passed after this
lifetime optimization.
The detector reconstruction head now folds its terminal `ConvTranspose(2x2,
stride 2) -> Add([1,C,1,1])` pairs into the transposed-convolution bias during
model load. This removes both final feature-map Add traversals for tiny, small,
and medium (confirmed as `ConvTranspose.0_bias` and `ConvTranspose.2_bias` in
the medium operator profile) while retaining exact output and all ISA paths.
The strict seven-image comparison and 11-image dimension/diversity stress
suite remained fully passing after the change.

The recognizer's direct `Conv -> Add([1,C,1,1]) -> ReLU` exports are now folded
at load time into the existing Conv+ReLU kernels. This adds the previously
missing no-Identity form of channel-bias folding, so small/medium stem
projections avoid a broadcast Add pass and its transient result. The same
matcher also recognizes canonical `Conv -> HardSigmoid -> Mul` gates and maps
them to the existing exact AVX-512/AVX2/NEON HardSwish kernel when encountered.
The default and forced-AVX2 tiny text-parity checks both passed 5/5 images, and
the 11-image 16x16-to-2560x1440 stress suite passed 11/11 after the change.

### Current optimization validation and comparison refresh

The current build adds the post-Conv Swish graph fusion described above. The
official medium model profiled here does not contain a matching
`Conv -> Sigmoid -> Mul` sequence after affine folding, so this generic fusion
does not contribute to the following current-model timing. The strict tiny
text-parity suite completed 5/5 images against ONNX Runtime after the change.
The benchmark below was then run on the same shared Windows host with
`PPOCR_BENCH_THREADS=16`, AVX-512 enabled dynamically, exact-Erf GELU, five
warmups and 20 measured iterations. It is recorded here as a reproducible
current-build comparison, not a portable capacity guarantee.

| Model | Pure C++ mean | ORT mean | C++ / ORT | Pure C++ FPS | ORT FPS |
|---|---:|---:|---:|---:|---:|
| tiny | 40.159 ms | 140.930 ms | 0.285x | 24.901 | 7.096 |
| small | 95.946 ms | 234.128 ms | 0.410x | 10.422 | 4.271 |
| medium | 342.556 ms | 599.235 ms | 0.572x | 2.919 | 1.669 |

For this run, native C++ latency was 71.5%, 59.0%, and 42.8% lower than ONNX
Runtime CPUExecutionProvider for tiny, small, and medium. Native throughput
was correspondingly 251.0%, 144.0%, and 74.9% higher. The native/ORT
median/p95 pairs were 39.217/44.732 vs. 142.425/199.114 ms (tiny),
94.667/103.429 vs. 234.956/276.820 ms (small), and 316.130/451.931 vs.
576.425/767.443 ms (medium). The shared Windows host is scheduling-sensitive,
so retain the ranges and rerun this paired procedure on a deployment CPU before
capacity planning. The existing tiny/small/medium repeated exact table remains
the longer-run model-size baseline.
### Dense-text stress result

The small-image result above is not representative of a screen with hundreds
of detected text regions. The native path groups equally sized crops into
bounded batches and schedules up to fourteen independent batches concurrently;
each batch retains its output order and uses an isolated execution workspace.
When multiple batches concurrently reach a CPU-parallel operator, one uses the
persistent worker set while contenders execute their independent channel split
locally. This prevents shared-pool serialization and nested-worker contention.
For the PP-OCR recognizer's terminal CTC head it also reduces logits directly
to argmax plus selected-character probabilities, avoiding an otherwise unused
`[batch,time,6906]` Softmax tensor.
Last-use elementwise activations (GELU, BatchNorm+GELU, ReLU, h-swish and
scale/shift) are updated in-place, eliminating transient activation copies. The
BatchNorm+GELU path now evaluates the exact-Erf GELU in the same pass as its
affine transform, removing a further full activation read/write in the default
exact mode. In the explicitly enabled approximate-GELU deployment mode, the
same affine result routes through AVX-512/AVX2 scale-shift plus vector GELU;
the corpus gates above validate that optional path.
Recognizer `Conv -> BatchNorm` inference pairs are folded into immutable
convolution weights and biases during model loading, removing the BatchNorm
activation pass without changing the deployed C++ runtime dependencies.
The detector FPN `2x2`, stride-2 transposed-convolution blocks have dedicated
AVX2 and AVX-512 implementations selected through the existing runtime ISA
dispatch. They vectorize independent input pixels while keeping the same
output/input-channel accumulation order; scalar and ARM builds retain the
portable implementation. In a direct dense-page operator profile, the two
blocks totalled 3.135 ms with AVX versus 3.597 ms scalar (about 12.8% for
those kernels).
The neighbouring detector reconstruction `2x2` valid convolutions now also
have dedicated AVX-512 and AVX2 kernels. They stream the four contiguous input
rows required by each output vector and avoid generic padding/interior checks;
ARM and scalar builds retain the portable kernel. On the current host this
changed the complete two-line small model from 110.364 ms to 98.112 ms and the
medium model from 352.690 ms to 338.791 ms in the same short-run configuration
(about 11.1% and 3.9% respectively), while dense tiny recognition measured
243.949 ms for 123 boxes.
The larger stride-one `3x3` detector layers use an AVX-512 four-output-channel
kernel as well. Each input vector is loaded once and accumulated into four
independent channel outputs, retaining the original channel/kernel accumulation
order. On the medium profile this reduced `Conv.121` from 26.809 ms to 5.910 ms
and `Conv.3` from 20.759 ms to 12.379 ms; the complete short-run medium OCR
measurement improved from 338.791 ms to 291.049 ms. AVX2, NEON, and scalar
paths remain available through runtime ISA dispatch.
The same four-output `3x3` algorithm is now implemented for AVX2 as well. An
explicit `PPOCR_DISABLE_AVX512=1` run exercised that fallback on the present
machine: medium completed in 404.166 ms and the strict tiny text corpus passed
7/7, confirming that advanced instructions remain dynamically optional rather
than a deployment requirement. ARM/NEON retains its portable vector path.
The scheduler now divides these fused `3x3` layers by four-channel groups,
instead of splitting a group between CPU workers. This preserves input-vector
reuse even for channel-parallel medium layers and safely sends only an incomplete
last group to the scalar tail. The resulting AVX-512 medium run measured
279.329 ms (3.580 FPS); strict text and the 11-image dimension/diversity corpus
both remained fully passing.
The same AVX-512 four-output strategy now covers the medium detector's square
stride-one `5x5` and `7x7` context layers. It cut the profiled `Conv.111`
7x7 from 8.472 ms to 3.898 ms and `Conv.114` 5x5 from 3.973 ms to 1.938 ms;
the complete medium result improved from 279.329 ms to 257.848 ms (3.878 FPS).
Unsupported ISAs and non-square/asymmetric kernels continue through the
existing AVX2/NEON/scalar paths.
Depthwise dispatch now routes square stride-one `7x7` and `9x9` kernels to the
existing AVX-512/AVX2 vector kernel too. This covers medium `Conv.64` (256
channels, 9x9): its local operator profile fell from 8.080 ms to 4.845 ms.
The native benchmark's current medium distribution remains sensitive to shared
host scheduling, so the paired end-to-end table is retained as the comparable
model-size result rather than replacing it with an unpaired one-off sample.
For `1x1 Conv -> ReLU` sequences, the AVX-512 path now applies the clamp before
storing each register-tiled output vector. This removes the separate activation
read/write for the fused detector projections while leaving AVX2, NEON, and
scalar execution on the existing safe Conv-then-ReLU path. The strict text and
11-image dimension/diversity gates passed after the change. On this shared
benchmark host the short medium samples remained scheduler-noisy, so no new
end-to-end speed claim is made without a paired repeated run.
The default exact-Erf GELU path is also parallelized across independent NCHW
planes through the persistent executor. BatchNorm+GELU now precomputes its
per-channel affine coefficients once and submits plane-local exact evaluation
without allocating another activation. On medium profiling this changed the
13 fused GELU calls from 26.743 ms to 7.297 ms while retaining default ONNX
Erf semantics; the strict text and 11-image stress gates passed with
`PPOCR_APPROX_GELU` unset. The optional approximate mode remains available for
deployment-specific validation, not as a correctness requirement.
The recognizer terminal CTC Top-1 reduction now parallelizes only across
independent batch sequences. Each sequence retains its own blank/repeated-token
state, while dense-page batches can distribute vocabulary scans and the selected
character confidence softmax over the persistent worker set. Exact decoded-text
and the 11-image dimension/diversity gates passed in default exact-GELU mode;
the shared-host dense benchmark remains variable, so its paired table remains
the authoritative capacity-planning reference.
CTC's 6,906-way per-step ArgMax also uses dynamically selected AVX-512 or AVX2
vector reductions. It explicitly scans the non-vector-width tail before
locating the first winning class, preserving the scalar tie rule and ensuring
the final vocabulary entries (including the space class) remain observable.
Both default AVX-512 and forced `PPOCR_DISABLE_AVX512=1` AVX2 runs passed the
strict 7-image text corpus; the current exact-GELU dense 123-box run measured
345.415 ms (median 343.977 ms, p95 359.279 ms), subject to the documented
shared-host scheduling variation.
AVX-512 four-output fusion now also covers stride-one asymmetric `1x3/1x5/1x7`
and `3x1/5x1/7x1` detector convolutions. It vectorizes the contiguous spatial
axis, preserves scalar border accumulation, and assigns four-channel groups to
workers so input loads are reused. Medium `Conv.112` (7x1) fell from a recent
3.363 ms profile to 1.211 ms. Default exact-GELU strict-text and 11-image
dimension/diversity gates remained passing; standalone medium timing is still
reported with the documented shared-host variance rather than as a new paired
ORT claim.
The DB connected-component postprocessor represents threshold and visited state
in one byte per probability-map pixel (`0` background, `1` pending, `2` visited),
instead of two full byte maps. This halves that postprocessor workspace while
leaving thresholding, traversal order, scores, boxes, and reading order unchanged.
The detector terminal `Sigmoid` is also executed in-place whenever its input
has no remaining consumers. It reuses the final DB-logit allocation as the
probability map, eliminating a peak activation allocation and copy while
preserving the exact scalar sigmoid expression.
Detector FPN nearest-neighbour integer upscales are implemented as direct
NCHW row replication, and recognizer inverse-Transpose pairs plus Conv channel
bias broadcasts are removed at model load. These changes eliminate dynamic
activation copies without relaxing the exact-GELU accuracy path.
The executor also prunes original constants made unreachable by load-time
Conv/BatchNorm folding. This applies to tiny, small, and medium without
changing runtime graph inputs. A fresh two-cycle mixed-size medium soak
(16x16, 3000x80, noisy scene and 1920x1080 UI page) used a 146.94 MiB
post-warmup baseline; the cycles were 147.21 and 148.18 MiB (+0.27 and
+1.24 MiB). The variation is Windows heap retention rather than monotonic
per-shape growth; longer soak runs should be used for release qualification.
The same mixed-size medium sequence in default exact-GELU mode used a 147.68
MiB post-warmup baseline; two cycles ended at 146.48 MiB and 152.16 MiB
(-1.20 / +4.48 MiB). This is bounded allocator retention rather than a
per-shape activation cache; use longer production soak runs for qualification.
A fresh four-cycle medium run after the pointwise-tile update used a 149.641
MiB post-warmup baseline and ended at 146.504 MiB (-3.137 MiB); its largest
cycle was only 2.022 MiB above baseline. The mixed sequence covered 16x16,
3000x80, low-contrast, and 1920x1080 dense-page inputs, again showing bounded
Windows allocator retention rather than persistent dynamic-shape growth.
The AVX2/AVX-512 direct-convolution path now also covers stride-1 7x7 and
asymmetric 3/5/7x1 or 1x3/5/7 detector kernels, while preserving the same
output/input-channel/kernel accumulation order; ARM and scalar paths retain
the portable implementation.
The dominant NCHW `1x1` convolution kernel also now computes four output
channels per shared SIMD input load (AVX-512, AVX2, and NEON implementations),
reducing hot-path input bandwidth while preserving each channel's accumulation
order.
The detector DB-head `2x2`, stride-1 `SAME_UPPER` max pool is likewise mapped
to a boundary-safe AVX-512/AVX2/NEON kernel, removing generic window and bounds
reconstruction from every output pixel.
The following repeatable stress measurement records the present result on
`build/ui_dense_1920x1080.ppm` (about 123--124 text regions):

| Engine | Warmup | Measured runs | Mean latency | Throughput |
|---|---:|---:|---:|---:|
| Pure C++ PP-OCRv6 | 4 | 10 | 233.805 ms | 4.277 FPS |
| ONNX Runtime 1.28 CPUExecutionProvider | 4 | 10 | 929.357 ms | 1.076 FPS |

The table uses one paired `compare.ps1` run with the persistent executor and
validated approximate-GELU setting. C++ is **74.8% lower latency** (0.252x) and
**297.5% higher throughput** in this specific run. It replaces the previous
serial-recognition baseline of 2680.808 ms, a **11.47x** native speedup.
A repeated parameter
sweep after the copy-elimination changes selected the current default of
`rec_batch_size = 4` and
`rec_parallelism = 14` for this 16-logical-CPU host; applications can tune
these options for their deployment CPU. Short dense runs can vary
materially with CPU scheduling, so use the command below and more runs on the
deployment target before capacity planning:

`Options::rec_batch_width_bucket` (or benchmark environment variable
`PPOCR_REC_WIDTH_BUCKET`) can additionally pack nearby natural crop widths
into one zero-right-padded batch; each crop keeps its own aspect-ratio resize.
The default remains `1`: a local 123-crop dense-page sweep did not demonstrate
a stable CPU benefit from padded work, so exact-width batching remains the
lower-memory default. Wider buckets are available for deployment-specific GPU
or batch-density tuning.

```powershell
powershell -ExecutionPolicy Bypass -File .\bench\compare.ps1 `
  -Image .\build\ui_dense_1920x1080.ppm -Runs 10 -Warmup 4 -Threads 16
```

The captured result is C++ 233.805 ms (median 232.980 ms, p95 244.485 ms) and
ONNX Runtime 929.357 ms (median 930.224 ms, p95 1098.989 ms). Keep repeated
paired measurement when refreshing the table, because dense-page latency can
vary with host scheduling.

The run used four warmups, 10 measured runs, and matching 16-thread CPU budgets.
Use longer repeated measurements on the deployment CPU before capacity planning;
dense-page latency also depends strongly on the detected crop mix.


### Expanded stress corpus

`bench/stress_regression.py` executes the native C++ pipeline over 11 deterministic
local stress images without adding a runtime dependency. It covers 16鑴?6 tiny text,
3000鑴?0 and 80鑴?000 strips, low contrast, rotation, blank input, dense synthetic
text, and 1920鑴?080 / 2560鑴?440 screenshots. The gate asserts bounded detection
counts, valid confidence/geometry, and high-confidence text anchors where exact
front-end parity is meaningful; rotated and density-only cases intentionally test
robustness rather than unstable line ordering.

```powershell
& 'D:\workprj\aicoder\.tmp\ocr-models\.venv\Scripts\python.exe' `
  .\bench\stress_regression.py `
  --cpp .\build\ppocr_demo.exe `
  --det D:\workprj\aicoder\.tmp\ocr-models\ppocrv6_tiny_det.onnx `
  --rec D:\workprj\aicoder\.tmp\ocr-models\ppocrv6_tiny_rec.onnx `
  --dict D:\workprj\aicoder\.tmp\ocr-models\dict_ppocrv6_tiny.txt `
  --images D:\workprj\aicoder\corelib\ocr\testdata\stress
```

Latest local run: **11/11 passed** (box counts: 123, 0, 1, 1, 20, 1, 3, 6, 743,
34, 38). This broadens the existing strict 7-image text-parity gate; it does not
weaken it.

### Recognition-regression corpus

`bench/accuracy_compare.py` compares complete C++ OCR results against an ONNX
Runtime CPU reference. Its default gate checks detected-line count and decoded
text; confidence and geometry are retained in the JSON report but do not gate
the test because the two standalone frontends have small resize/crop rounding
differences. Pass `--strict-geometry` to also gate those fields.

`bench/accuracy_expectations.json` supplies independently recorded expected
text for the deterministic corpus, so the same run also checks C++ results
against known text rather than only comparing two inference backends.

The latest 7-image corpus passed **7/7 text-parity cases**. It includes clean
multi-line English, a single word, a 3000-pixel-wide receipt line, a
low-contrast/noisy scene, a blank image, and a 16x16 tiny-text case. The
samples are existing deterministic test assets from
`D:\workprj\aicoder\corelib\ocr\testdata`.

An additional multilingual and extreme-shape sweep covers Chinese and mixed
language text, a 80x3000 vertical strip, rotated text, synthetic dense blobs,
and 1920x1080/2560x1440 screenshots. Chinese/mixed baseline samples retain
exact decoded-text parity. The pathological vertical/dense screenshot samples
remain diagnostic rather than release gates because detector/crop rounding can
alter weak low-confidence regions between the two independently implemented
frontends. `accuracy_compare.py` now forces UTF-8 report output so this corpus
can be run from a Windows GBK console without losing dictionary characters.

```powershell
$images = @(
  'D:\workprj\aicoder\corelib\ocr\testdata\en.png',
  'D:\workprj\aicoder\corelib\ocr\testdata\line_en.png',
  'D:\workprj\aicoder\corelib\ocr\testdata\stress\single_word.png',
  'D:\workprj\aicoder\corelib\ocr\testdata\stress\wide_strip_3000x80.png',
  'D:\workprj\aicoder\corelib\ocr\testdata\stress\noisy_lowcontrast.png',
  'D:\workprj\aicoder\corelib\ocr\testdata\stress\blank_800x600.png',
  'D:\workprj\aicoder\corelib\ocr\testdata\stress\tiny_16x16.png'
)
& 'D:\workprj\aicoder\.tmp\ocr-models\.venv\Scripts\python.exe' `
  .\bench\accuracy_compare.py `
  --cpp .\build\ppocr_demo.exe `
  --python 'D:\workprj\aicoder\.tmp\ocr-models\.venv\Scripts\python.exe' `
  --det 'D:\workprj\aicoder\.tmp\ocr-models\ppocrv6_tiny_det.onnx' `
  --rec 'D:\workprj\aicoder\.tmp\ocr-models\ppocrv6_tiny_rec.onnx' `
  --dict 'D:\workprj\aicoder\.tmp\ocr-models\dict_ppocrv6_tiny.txt' `
  --expectations .\bench\accuracy_expectations.json `
  --images $images
```

Captured report: `build/accuracy_ground_truth_7_ctctop1_sparse.json` (7/7
passed after batch/concurrent recognition and terminal-CTC reduction).

### Dynamic-size memory soak

`ppocr_mem_soak` keeps one C++ OCR instance alive while repeatedly alternating
tiny, wide-strip, noisy, and 1920x1080 UI images. It reports process-private
bytes after every complete mixed-size cycle. Intermediate tensors are released
after their final graph consumer, model weights remain immutable, and no
per-shape activation cache is retained. Detector results (including the large
probability map) are additionally scoped through DB post-processing and released
before concurrent recognition batches begin, so the allocator can reuse their
storage rather than combining detector and recognizer activation peaks.

The same process-level workload can be run against ONNX Runtime with
`bench/ort_mem_soak.py`. On this Windows host, using 12 mixed-size cycles:

| Engine | Baseline private bytes | Final private bytes | Final change |
|---|---:|---:|---:|
| Pure C++ PP-OCRv6 | 23.825 MiB | 24.633 MiB | +0.809 MiB |
| ONNX Runtime CPU | 145.371 MiB | 203.805 MiB | +58.434 MiB |

The current C++ run ranged from 22.735 MiB to 27.703 MiB after the baseline;
its final working set was 0.809 MiB above the immediate post-load baseline,
without a monotonic per-cycle increase. ONNX Runtime retained roughly 58 MiB
in the prior corresponding run. This is a process-private-memory measurement
for the specified machine and models, not a portable memory guarantee; rerun
it on the deployment target.

Build PPM copies for the native soak from the same PNG source files:

```powershell
$code = @'
from PIL import Image
from pathlib import Path
source = Path(r'D:\workprj\aicoder\corelib\ocr\testdata')
output = Path(r'.\build\mem_samples'); output.mkdir(exist_ok=True)
for name in ['stress/tiny_16x16.png', 'stress/wide_strip_3000x80.png',
             'stress/noisy_lowcontrast.png', 'stress/ui_dense_1920x1080.png']:
    Image.open(source / name).convert('RGB').save(output / (Path(name).stem + '.ppm'), format='PPM')
'@
$code | & 'D:\workprj\aicoder\.tmp\ocr-models\.venv\Scripts\python.exe' -
```

Then run the C++ side (replace the model paths if needed):

```powershell
.\build\ppocr_mem_soak.exe <det.onnx> <rec.onnx> <dict.txt> 12 `
  .\build\mem_samples\tiny_16x16.ppm `
  .\build\mem_samples\wide_strip_3000x80.ppm `
  .\build\mem_samples\noisy_lowcontrast.ppm `
  .\build\mem_samples\ui_dense_1920x1080.ppm
```

### Resolution ladder and post-large-image memory recovery

For a stricter retention check, generate a deterministic page ladder that
climbs from `16x16` through `3840x2160`, includes ultra-wide and ultra-tall
pages, and ends with another `64x64` page:

```powershell
& 'D:\workprj\aicoder\.tmp\ocr-models\.venv\Scripts\python.exe' `
  .\bench\generate_memory_ladder.py --output .\build\memory_ladder

.\build\ppocr_mem_soak.exe <det.onnx> <rec.onnx> <dict.txt> 2 `
  .\build\memory_ladder\ladder_16x16.ppm `
  .\build\memory_ladder\ladder_64x64.ppm `
  .\build\memory_ladder\ladder_160x120.ppm `
  .\build\memory_ladder\ladder_320x240.ppm `
  .\build\memory_ladder\ladder_640x480.ppm `
  .\build\memory_ladder\ladder_1280x720.ppm `
  .\build\memory_ladder\ladder_4096x256.ppm `
  .\build\memory_ladder\ladder_256x4096.ppm `
  .\build\memory_ladder\ladder_1920x1080.ppm `
  .\build\memory_ladder\ladder_2560x1440.ppm `
  .\build\memory_ladder\ladder_3000x1600.ppm `
  .\build\memory_ladder\ladder_3840x2160.ppm `
  .\build\memory_ladder\ladder_recovery_1920x1080.ppm `
  .\build\memory_ladder\ladder_recovery_4096x256.ppm `
  .\build\memory_ladder\ladder_recovery_256x4096.ppm `
  .\build\memory_ladder\ladder_return_64x64.ppm
```

For CI or a repeatable pass/fail report, run the bounded-recovery checker. It
records every resolution and requires the final `64x64` sample to remain
within 20% of both the warmed baseline and the final ultra-tall recovery
sample, as well as within 20% of the earlier `3840x2160` sample. The report
includes `declined_after_largest`, so allocator capacity retention
cannot be confused with an actual post-large-image recovery observation:

```powershell
& 'D:\workprj\aicoder\.tmp\ocr-models\.venv\Scripts\python.exe' `
  .\bench\memory_ladder_regression.py `
  --cpp .\build\ppocr_mem_soak.exe --det <det.onnx> --rec <rec.onnx> `
  --dict <dict.txt> --images .\build\memory_ladder --cycles 2 --backend cpu `
  --require-post-large-decline
```

`ppocr_mem_soak` loads, infers, and destroys one RGB image at a time and emits
one private-byte sample after every resolution.  The final small-page sample
is therefore the direct recovery observation after the staged 4K and
extreme-aspect-ratio tail; compare it with the `ladder_3840x2160` and final
`ladder_recovery_256x4096` samples and the next cycle rather than treating
the process allocator's high-water capacity as a leak.  Run the same command
with `PPOCR_BACKEND=hybrid` to include the persistent Vulkan buffers in the
process-level measurement.

The current generator intentionally uses an ascending, aspect-ratio-diverse
16-sample sequence: `16x16 -> 64x64 -> 160x120 -> 320x240 -> 640x480 ->
1280x720 -> 4096x256 -> 256x4096 -> 1920x1080 -> 2560x1440 -> 3000x1600 ->
3840x2160 -> 1920x1080 -> 4096x256 -> 256x4096 -> 64x64`. The final staged
recovery tail checks that memory remains bounded after both 4K area and
extreme-aspect-ratio images. In addition to private bytes,
the native JSON now includes Windows working-set bytes for each measured
sample.  Private bytes remain the bounded-recovery gate; working set is
reported as complementary residency evidence because the Windows heap and
Vulkan driver may retain reusable committed capacity after the page and its
activations have been destroyed.

The generator also includes native **5K** (`5120x2880`) and **8K**
(`7680x4320`) source pages.  The 8K sample is followed immediately by a 4K
recovery sample, so use `--require-post-large-decline` to make both Windows
private bytes and working set decline at that exact transition a strict gate.
This catches source-image retention even though detector inference itself is
normally capped by `det_limit_side_len`.

### 2026-08-14: current small-to-large image-memory check

The checked-in generator was rerun against the current medium PP-OCRv6 CPU
build.  It creates the deterministic 16-image sequence above, processes each
image with one long-lived OCR instance, and destroys the decoded RGB image
before sampling process memory.  The result therefore measures retained
runtime/allocator capacity rather than intentionally retained source images.

In the current one-cycle run, private memory was **147,017,728 B** at
`3840x2160`, then immediately declined to **146,296,832 B** at the following
`1920x1080` sample (**-720,896 B**, **-0.49%**).  Working set also declined
from **150,679,552 B** to **150,085,632 B** (**-593,920 B**).  After the
complete recovery tail, the final `64x64` sample was **147,124,224 B** private
and **151,035,904 B** working-set memory: below the pre-return peak
(**148,320,256 B** private / **152,371,200 B** working set), and only **2.21%**
above the warmed private baseline (**143,937,536 B**).  The bounded-recovery
gate passed.  Full per-resolution evidence is
`build/memory_ladder_medium_cpu_current_20260814.json`.

Windows allocation is intentionally treated as a bounded-recovery result, not
as a requirement that every allocator page be decommitted immediately: the
explicit first-post-4K samples prove the observed downward transition, while
the final small-image sample guards the durable footprint after the whole
large-image sequence.

### 2026-08-13: current ascending-resolution memory measurement

With the current default recognizer batch size of 1, CPU backend, 16 kernel
workers, and approximate GELU, medium PP-OCRv6 passed a fresh one-cycle
ascending-resolution check. The run covers every generated image from
`16x16` through `3840x2160` (including `4096x256` and `256x4096`), then the
staged recovery tail and a final `64x64`. Private bytes were **147,234,816 B**
after `3840x2160` and **147,144,704 B** after the final `64x64`, a decline of
**90,112 B (0.06%)**. Working set likewise declined from **151,109,632 B** to
**150,913,024 B** (**196,608 B**). The final private-byte value was **0.008%**
above the fully warmed **147,132,416 B** baseline and 2.38% below the
pre-return peak, so every 20% bounded-recovery gate passed. The complete
per-resolution record is
`build/memory_ladder_medium_cpu_ascending_recheck_20260813.json`; values are
host-specific allocator observations, rather than a promise of immediate heap
decommit on every operating system.

### 2026-08-13: per-resolution memory-record validation

The ladder checker now preserves the private-byte and working-set sample for
**every** image resolution in its JSON report, together with the cycle peak
and a direct post-`2560x1440` recovery flag.  This makes the check suitable for
both leak detection and deployment profiling: a final small-page allocation is
not mistaken for the large-page allocation just because a heap or driver keeps
reusable capacity committed.

On the current AVX-512 Windows host, medium PP-OCRv6 with 16 kernel workers
and opt-in approximate GELU passed two fresh CPU ladder cycles.  The warmed
private baseline was **146,632,704 bytes**.  After `2560x1440`, the final
`64x64` sample was **148,533,248 bytes** in cycle 1 (+1.30% baseline, +2.00%
from the largest-page sample) and **143,220,736 bytes** in cycle 2 (-2.33%
baseline, -1.23% from the largest-page sample).  Thus direct private-byte
recovery occurred in the second cycle but not the first: Windows heap reuse
can retain or move committed capacity between samples.  Both final samples
remained within the 20% bounded-recovery limit, which is the pass/fail gate;
the JSON deliberately retains the explicit per-cycle recovery flag instead of
claiming unconditional decommit after a large image.

The same two-cycle Hybrid Vulkan run also passed.  Its warmed private baseline
was **394,969,088 bytes** and each final `64x64` sample was **396,017,664
bytes** (+0.265%).  The final private bytes equalled the immediately preceding
`2560x1440` sample in both cycles, while the working set was essentially flat
(+12,288 bytes in cycle 1 and +8,192 bytes in cycle 2).  This is persistent
Vulkan/Windows allocator capacity rather than evidence that an image or its
activations remain live.  These are machine-specific process measurements;
the exact per-resolution values are retained in
`build/memory_ladder_medium_{cpu,hybrid}_20260813.json` when the commands are
run locally.

On 2026-08-12, the current medium model passed this expanded two-cycle gate.
CPU warmed at **148,127,744 B** and returned to **147,148,800 B** and
**147,152,896 B** (both **-0.66%** of baseline); the final small-page samples
were 2.33% and 1.70% above their immediately preceding largest-page samples.
Hybrid Vulkan warmed at **393,490,432 B** and returned at **395,317,248 B**
(+0.46%) and **395,341,824 B** (+0.47%).  All private-byte deltas passed the
20% gate.  The resident working-set observations are also captured in the
checker output rather than being inferred from private bytes; on this shared
Windows host they remained within the allocator/driver's warmed range, not a
claim that either heap must promptly decommit pages.

On 2026-08-12, the tiny model with 16 CPU workers was run through this exact
seven-image ladder.  CPU-only warmed to **9,400,320 bytes**; after the largest
`2560x1440` page its next small-page sample was **9,465,856 bytes**
(**+65,536 bytes**, 0.7% of baseline).  Hybrid warmed to **116,305,920 bytes**
and the corresponding post-large small-page sample was **115,802,112 bytes**
(**-503,808 bytes**).  These per-image values show recovery after the largest
input for both modes.  They are Windows private-byte observations on this
host, not a cross-platform leak proof or a fixed memory guarantee.

The same seven-page gate was rerun after the depthwise-pointwise-GELU graph
fusion with the **medium** PP-OCRv6 model (two measured cycles, default bounded
CPU scheduler).  Both modes passed the 20% bounded-recovery limits:

| Backend | Warmed baseline | Final `64x64`, cycle 1 | Final `64x64`, cycle 2 | Result after `2560x1440` |
|---|---:|---:|---:|---|
| CPU | 136.351 MiB | 140.019 MiB (+2.69%) | 140.008 MiB (+2.68%) | Allocator private bytes did not immediately decline, but remained stable and 2.24閳?.29% above the largest-page sample. |
| Hybrid Vulkan | 360.574 MiB | 361.582 MiB (+0.28%) | 361.582 MiB (+0.28%) | Equal to the immediately preceding largest-page sample in both cycles. |

The CPU observation is expected on Windows: completed image and activation
objects have been destroyed, while the process allocator may retain reusable
blocks.  Consequently the test reports both `declined_after_largest` and the
bounded deltas instead of treating an unchanged private-byte high-water mark
as proof of a live-image or tensor leak.  The Hybrid samples include its
persistent Vulkan allocations.

The same CPU-only one-cycle ladder was also run for the other two PP-OCRv6
sizes: small warmed to **37,662,720 bytes** and returned at
**38,973,440 bytes** after its largest page (**+1,310,720 bytes**); medium
warmed to **147,435,520 bytes** and returned at **148,439,040 bytes**
(**+1,003,520 bytes**).  In both cases the return sample remained within 3.5%
and 0.7% of their post-warm-up baselines respectively, with no retained RGB
source images between calls.

The reusable checker was then run for two full ladders after the per-image
sampler change.  All three CPU models passed the 20% bounded-recovery gate:
tiny returned within **+1.14%**, small within **+10.09%**, and medium within
**+2.46%** of their warmed baselines; each final `64x64` sample was also at or
below the pre-large `1920x1080` sample except medium's first cycle (**+0.81%**).
Tiny hybrid also passed, returning within **+0.08%** of baseline, including
the persistent Vulkan allocation.  These figures are deliberately a bounded
allocator-reuse test, not an assertion that Windows must immediately release
committed heap pages to the OS.

The current checker also makes the immediate recovery result explicit.  On a
fresh two-cycle tiny run with the default 12 workers, CPU warmed to
**10,047,488 bytes** and returned at **9,379,840** then **9,441,280 bytes**;
both cycles reported `declined_after_largest=true` (2.97% and 9.64% below the
preceding `2560x1440` samples).  Hybrid warmed to **135,745,536 bytes** and
returned at **135,098,368** then **135,102,464 bytes**, both below its warmed
baseline but **not** immediately below the preceding largest-image samples
(`declined_after_largest=false`, +1.24% and +0.63%).  This distinction is
intentional: the test reports the actual post-large behavior and separately
enforces bounded recovery, rather than incorrectly claiming immediate heap
release for persistent Vulkan/Windows allocator capacity.

### 2026-08-12: current ascending-resolution memory check

The deterministic seven-image ladder was regenerated and rerun on the current
binary: `64x64 -> 320x240 -> 640x480 -> 1280x720 -> 1920x1080 -> 2560x1440
-> 64x64`.  Each sample performs `load -> Recognize -> destroy ->
PrivateBytes`, so the reported process-private bytes include runtime working
memory but do not retain the source RGB page between samples.  The bounded
20% return gate passed in every run below.

| Model / backend | Warmed baseline | Cycle 1 final 64x64 | Cycle 2 final 64x64 | Post-largest result |
|---|---:|---:|---:|---|
| tiny / CPU | 9,367,552 B | 9,412,608 B (+0.48%) | 9,424,896 B (+0.61%) | Fell below the preceding 2560x1440 sample in both cycles (11.17%, 1.75%). |
| medium / CPU | 148,254,720 B | 146,284,544 B (-1.33%) | 147,300,352 B (-0.64%) | Bounded return; final sample was 1.08% / 2.65% above the immediately preceding large sample. |
| tiny / hybrid Vulkan | 115,851,264 B | 99,143,680 B (-14.42%) | 98,140,160 B (-15.29%) | Bounded return; final sample was 0.004% / 0.017% above the immediately preceding large sample. |

`declined_after_largest` is therefore true for the CPU tiny run and false for
the medium CPU and hybrid runs.  That false value is an observation, not a
leak verdict: Windows heap and persistent Vulkan allocations may retain or
rearrange reusable capacity.  The two complementary checks閳ユ敄he final-small
sample versus its post-warm-up baseline and the stated largest-page
comparison閳ユ攷ake the memory behavior auditable without claiming that a process
must immediately return committed pages to the OS.

After the subsequent depthwise-pointwise-GELU fusion, the medium ladder was
rerun with two cycles.  CPU warmed at **142,974,976 B**; its final `64x64`
samples were **146,821,120 B** (+2.69%) and **146,808,832 B** (+2.68%).
They were 2.29% and 2.24% above the respective `2560x1440` samples, so
`declined_after_largest=false`, yet both are stable and well within the 20%
bounded-return gate.  Hybrid warmed at **378,089,472 B** and returned at
**379,146,240 B** in both cycles (+0.28%); each final small image equalled its
preceding largest-image sample (`declined_after_largest=true`).  Thus the
test continues to distinguish released OCR image/tensor lifetimes from normal
Windows heap or persistent Vulkan capacity retention.

### 2026-08-12: admitted depthwise-to-pointwise Vulkan segment

The graph loader now recognizes an exact `depthwise Conv(bias) -> 1x1
pointwise Conv(bias)` pair and emits `FusedDepthwisePointwiseConv`.  Its CPU
path remains the same two SIMD kernels, while hybrid can submit the existing
Vulkan chain in one command buffer: the depthwise result stays in a device
storage buffer and only the chain boundaries transfer.  Admission is cached
per full NCHW geometry and accepts GPU only after a repeated H2D + dispatch +
fence + D2H comparison and elementwise CPU-result validation show that the
complete segment is no slower.  Consequently this reduces CPU occupancy when
the Radeon/driver wins without forcing a transfer-bound GPU route.

The feature covers the detector MobileNet backbone pairs in all three model
sizes.  AVX-512 kernel smoke and the four diverse decoded-text comparisons
for tiny, small and medium all passed (**12/12**).  The current Radeon 780M
primitive probe measures a representative depthwise-to-pointwise batch at
**0.76714 ms GPU vs 0.20474 ms CPU**, so this particular UMA shape is
correctly rejected; it is evidence that the new graph route retains the
no-slower policy rather than a claim of blanket GPU acceleration.  A current
20-run CPU-only `en.ppm` snapshot (16 kernel workers, approximate GELU)
measured tiny **27.954 ms**, small **60.455 ms**, and medium **195.588 ms**.
`gpu_only` still fails explicitly because the complete PP-OCRv6 graph,
post-processing and CTC decode are not yet device resident; it never silently
falls back to CPU.

### 2026-08-12: depthwise-to-pointwise CPU graph fusion

The same strict MobileNet `depthwise Conv(bias) -> 1x1 pointwise Conv(bias)`
pattern now executes as one graph node on CPU too.  It removes the executor's
hash-map entry, node dispatch and intermediate tensor lifetime from the graph;
the internal temporary remains only for the two exact SIMD kernels.  This is
an intentional graph/lifetime fusion, not a mathematical approximation.  Set
`PPOCR_DISABLE_DEPTHWISE_POINTWISE_FUSION=1` for an A/B fallback.

On the current 16-worker AVX-512 host, three warm-ups and fifteen `en.ppm`
runs showed the fused path at **26.784 / 56.596 / 174.962 ms** for tiny,
small and medium, respectively, compared with **27.121 / 58.209 / 228.572
ms** with the graph fusion disabled.  The medium result is host-scheduling
sensitive, so the reproducible conclusion is the consistent no-regression
and gains for tiny/small, while deployment owners should rerun the medium A/B
on their target CPU. A separate 20-run medium A/B measured **213.711 ms**
fused versus **222.719 ms** disabled.  The exact decoded-text gate passed
**12/12** across tiny/small/medium and four diverse inputs.  A two-cycle
medium resolution ladder also passed the bounded memory-recovery gate:
147,255,296-byte warmed baseline; final small-page samples were -0.68% and
-0.08% from baseline.  As elsewhere, the final samples did not immediately
fall below the preceding maximum-size sample, which is normal reusable
Windows allocator capacity rather than a leak assertion.

### 2026-08-12: depthwise-pointwise-GELU liveness fusion

The loader now extends that exact single-use pattern through an immediately
following fused GELU as `FusedDepthwisePointwiseConvGelu`.  The pointwise
output is activated in place, rather than re-entering the generic graph loop
to rename and transform it.  It preserves FP32 operation order (the same
existing GELU kernel runs after the two convolutions) and keeps the Vulkan
segment's no-slower admission boundary intact; a selected GPU segment still
returns before the exact CPU GELU pass.  The targeted A/B escape hatch is
`PPOCR_DISABLE_DEPTHWISE_POINTWISE_GELU_FUSION=1`.

The AVX-512 kernel smoke and the 12/12 decoded-text ONNX Runtime gate passed.
On this shared host, a 15-run `en.ppm` A/B did not establish a uniform
latency benefit: tiny was 29.137 vs 28.908 ms, small 62.292 vs 62.007 ms,
and medium 205.199 vs 221.369 ms (fusion enabled vs disabled).  It is retained
because it removes a graph-lifetime boundary and shows a medium improvement,
but the table intentionally does not promote it to a universal CPU speedup.

### 2026-08-12: expanded batched ordinary-Conv SIMD coverage

`Conv2dBatch()` now carries the batch index through the existing runtime-gated
SIMD kernels for valid `2x2`, padded `5x5`/`7x7`, and asymmetric
`1x3`/`3x1`/`1x5`/`5x1` stride-one convolutions, in addition to the prior
`3x3` stride-one/two paths.  This removes one worker-pool dispatch per crop
for the medium detector's context layers while preserving each output's
increasing-channel/kernel accumulation order.  AVX-512 is selected only after
the existing CPUID/XCR0 guard; AVX2 and portable/NEON paths retain their safe
fallback behavior.

The low-level smoke now compares N=3 batched output against three independent
calls for all of those shapes and ReLU tails.  It passed with normal AVX-512
dispatch and with `PPOCR_DISABLE_AVX512=1` (AVX2 fallback).  A post-change
four-identical-page medium CPU batch test (`det_batch_size=4`,
`rec_batch_size=4`, three measured runs) was result-identical and measured
**793.299 ms** serial versus **710.437 ms** batched (**1.117x**).  The same
tiny hybrid configuration was result-identical and measured **126.750 ms**
serial versus **85.669 ms** batched (**1.480x**).  These are local scheduler
measurements, not a universal Vulkan claim: the concurrent Vulkan primitive
smoke continued to select GPU only for its no-slower affine+Swish probe and
retained CPU for the transfer-bound Conv and GEMM probes.

The full small-to-large-to-small CPU ladder was rerun for all model sizes with
the expanded kernels.  All passed the three 20% bounded-recovery limits.
Observed final-small change from the post-warm baseline was tiny **+1.72%**
and **+8.51%**, small **-1.57%** and **+8.90%**, and medium **+0.70%** and
**+0.02%** over two cycles.  `declined_after_largest` is retained as a fact
rather than a pass condition, because Windows heap trimming is nondeterministic
even when the runtime has released the input and activation vectors.

### 2026-08-12: default AVX-512 wide-context Conv tile

The AVX-512 eight-output tile for the PP-OCRv6 medium detector's `5x5` and
`7x7` context layers is now enabled by default for its narrow, runtime-checked
shape (`C_in >= 32`, `C_out >= 8`, and both spatial dimensions at least 32).
It reuses each input vector across eight output reductions; smaller shapes,
AVX2, NEON, and scalar targets retain their existing paths. Set
`PPOCR_DISABLE_AVX512_CONV_WIDE8=1` for a deployment-local A/B test.

On the local AVX-512 host with 12 workers, three warm-ups and twelve `en.ppm`
runs, medium CPU measured **424.234 ms** with the tile and **659.937 ms** with
`PPOCR_DISABLE_AVX512_CONV_WIDE8=1` (same model/input; shared-host variation
is reflected by the recorded medians **404.622 / 668.809 ms**).  Fresh
ONNX-Runtime decoded-text comparison remained **12/12** across tiny, small,
and medium on `16x16`, English, low-contrast, and `3000x80` inputs. The medium
two-cycle `64x64 閳?2560x1440 閳?64x64` ladder also passed all bounded-memory
checks, ending at **+0.19%** and **+2.72%** from its warmed baseline. Vulkan
primitive smoke remained numerically passing; hybrid selected only its
measured no-slower affine+Swish and depthwise+Swish samples in that run.

### 2026-08-12: medium worker and transpose-tile recheck

A fresh worker sweep after the wide-context default confirms the library's
automatic **16-worker** ceiling remains appropriate for the broad medium
graph, while tiny/small retain their shorter-kernel behavior: on the same
`en.ppm` setup, medium measured **191.664 / 205.187 / 185.906 ms** with
8/12/16 workers; tiny measured **28.434 / 28.797 ms** and small
**57.390 / 59.147 ms** at 12/16 workers.  The 16-worker medium CPU retest
(3 warm-ups, 20 runs) averaged **185.325 ms**; paired ONNX Runtime CPU on the
same image/run count averaged **260.194 ms**, a host-specific **28.8%** lower
native latency.

The AVX-512 four-output `ConvTranspose2x2` FPN experiment was rechecked too.
It measured **186.232 ms** versus **181.971 ms** for the established
single-output implementation (3 warm-ups, 10 runs), so it remains opt-in via
`PPOCR_ENABLE_AVX512_TRANSPOSE2X2_TILE4=1`; it is not promoted merely because
it reduces some input loads. This preserves the faster default and avoids a
regression on the current target.

### 2026-08-12: device-resident depthwise/pointwise Vulkan segment

The Vulkan backend now has a numerically checked two-dispatch MobileNet
`depthwise Conv -> 1x1 pointwise Conv` primitive.  It uploads the chain input
once, leaves the depthwise feature map in a mapped Vulkan storage buffer,
inserts an explicit compute shader write-to-read memory barrier, and reads
back only the pointwise result.  Its hybrid admission measures that complete
boundary against the corresponding CPU `DepthwiseConvBatch` plus
`PointwiseConvBatch` pair and rejects Vulkan when it is slower.

On the local Radeon 780M / AVX-512 host, the odd-shape batched numerical smoke
passed, but the representative `N=4, C=32, M=40, 31x29, 3x3` full-boundary
probe measured **0.836 ms Vulkan vs 0.271 ms CPU**.  Hybrid therefore keeps
the CPU path for this segment.  The primitive is deliberately not folded into
the production ONNX graph yet: its current model-level floating-point drift
needs a decoded-text tolerance gate before any performance claim or default
dispatch change.  This preserves exact production behavior while retaining a
real, barrier-validated GPU building block for subsequent fully-resident
segments.

The batch smoke executable now also prints its individual timed samples, so a
throughput claim can be audited for scheduler outliers rather than relying on
one aggregate number.  With four equal `en.ppm` pages, batch size four, 16 CPU
kernel workers, two warm-ups and eight timed runs, CPU serial閳妼atch results
were **120.826閳?5.704 ms (1.410x)** for tiny, **281.621閳?27.853 ms (1.236x)**
for small, and **859.114閳?44.174 ms (1.018x)** for medium. Hybrid under the
same batch settings measured **166.786閳?08.215 ms (1.541x)** tiny,
**331.511閳?51.555 ms (1.318x)** small, and **824.029閳?49.212 ms (1.100x)**
medium. Serial/batch OCR outputs were byte-for-byte equivalent in each run.
The hybrid figures are page-batching capacity gains; they do not claim that
the partial Vulkan graph is faster than CPU for every single page.

A fresh 12-cycle run after this lifetime change used a 23.488 MiB baseline and
finished at 26.664 MiB (+3.176 MiB); its cycle deltas ranged from -0.543 MiB to
+4.055 MiB with no monotonic growth. Windows heap retention makes the final
private-byte delta workload- and allocator-dependent, so the important release
criterion is the non-growing mixed-shape sequence, not a single final number.

After the batched constant-weight MatMul change, a medium-model two-cycle
mixed-size check covering `16x16`, `3000x80`, `80x3000`, and the 119-box
`1920x1080` UI page used **293,961,728 bytes** after warm-up; the two cycle
measurements were **811,008 bytes below** and **471,040 bytes below** that
baseline. This verifies that flattening a batch is a scheduling change, not a
per-shape tensor cache or an activation-retention increase. As above, it is a
Windows allocator-capacity observation rather than a universal leak proof.

After adding the Vulkan channel-affine batch buffers, the same medium mixed
sequence was rerun for two cycles with the default hybrid policy. Its
post-warm-up baseline was **290,508,800 bytes**; cycles finished at
**+5,214,208** and **+2,506,752 bytes**. The high-water movement is bounded
Windows allocator retention after the large UI page, not a per-cycle growth
trend; GPU affine admission was rejected on this host, so the extra persistent
Vulkan coefficient storage was not exercised by the CPU execution path.

After the BatchNorm-Swish in-place activation update, a fresh CPU medium
mixed-size soak (`16x16`, `3000x80`, `80x3000`, and the dense `1920x1080` UI
page) used **286,793,728 bytes** post-warm-up. Its next two cycles were
**9,101,312** and **5,144,576 bytes below** baseline. This is a bounded
allocator observation on Windows, but it confirms the source-buffer reuse does
not retain changing-shape recognition activations.

The captured C++ and ONNX Runtime reports are
`build/memory_soak_cpp_12.json` and `build/memory_soak_ort_12.json`. A
post-optimization 6-cycle C++ check is also retained as
`build/memory_soak_cpp_parallel4_6.json`: it finished 1.034 MiB below its
24.629 MiB post-warmup baseline, demonstrating that bounded concurrent
recognition does not retain every dynamic input shape.

### 2026-08-12: cross-page recognition batching

`OCR::RecognizeBatch` now keeps dynamic detector resizing per page, then
coalesces equal-width crops from all submitted pages into bounded NCHW
recognizer batches. This is useful for both CPU SIMD and the Vulkan batch-Y
dimension; it does not pad unrelated detector pages, does not change result
order, and releases each detector probability map before recognition starts.

On this Radeon 780M / Windows host, medium PP-OCRv6, four copies of `en.ppm`,
`rec_batch_size=8`, one recognizer worker and one detector worker:

| Path | Serial pages | Cross-page batched | Speedup |
|---|---:|---:|---:|
| CPU | 773.012 ms | 744.098 ms | 1.039x |
| Hybrid | 854.746 ms | 1249.711 ms | 0.684x |

The batch smoke test confirms exact public `Result` equivalence with serial
inference. Hybrid is intentionally not presented as a win here: the current
UMA Vulkan implementation admits each segment only when its complete transfer
+ dispatch + readback path is no slower than CPU, while the coalesced workload
still has significant CPU-only graph work. Four-image mixed-size CPU coverage
(`en`, low-contrast, wide strip, dense 1920x1080 UI; 128 results) also passed
the serial-result check at 1.004x. A tiny-model four-page CPU check measured
143.472 ms serial and 139.913 ms batched (1.025x). Re-run these workloads on
the deployment hardware; they are measurements, not portability claims.

### 2026-08-12: depthwise Vulkan batch kernel and hybrid batch gate

The Vulkan depthwise-convolution path now maps one workgroup to one NCHW
channel and up to 1,024 output pixels. Its `[KH,KW]` filter is cooperatively
staged in workgroup memory (up to `16x16`), while Vulkan's Y dimension remains
the independent crop-batch dimension. PP-OCRv6's MobileNet-style depthwise
filters therefore avoid reloading the same channel filter for every output
pixel. Larger exported filters safely use the existing global-memory fallback.

The Radeon 780M numerical smoke test passed for a padded `N=2, C=5, 7x9,
3x3` depthwise batch. Its latest complete synchronous measurement was
**0.44340 ms GPU / 0.35062 ms CPU**, so the strict H2D + dispatch + D2H
admission policy correctly retained the CPU SIMD path for that shape. This is
an optimized, batch-capable GPU primitive, not a claim that the incomplete
GPU-resident OCR graph is faster on this UMA GPU.

The same build also ran the public batch-equivalence harness on tiny PP-OCRv6
with `rec_batch_size=8`, one detector worker and one recognizer worker across
four varied pages (`en`, low-contrast, `3000x80`, dense `1920x1080`). All
**129** decoded results matched serial `Recognize()` exactly. The one-shot
local measurement was **1081.928 ms serial / 1167.667 ms batched** for CPU and
**1105.010 ms serial / 1075.921 ms batched** for hybrid. These scheduler
samples are intentionally retained as measurements rather than a universal
throughput promise; the dense mixed-width workload has limited equal-width
crop coalescing. A separate CPU-versus-hybrid dense-page result comparison was
also byte-for-byte identical.

### 2026-08-12: cross-crop CPU pointwise scheduling

The CPU 1x1-convolution path now accepts a real NCHW batch in one scheduling
call. On x86 it flattens `(crop N, output-channel tile)` into the persistent
AVX-512/AVX2 worker queue, retaining the existing per-channel FP32 reduction
order and ISA-specific four/eight-output vector kernels. ARM/NEON and scalar
builds retain their portable per-crop path. This removes repeated worker-pool
barriers for batchable recognizer projections without changing the model graph
or requiring ONNX Runtime.

The strict CPU-versus-ONNX Runtime decoded-text gate passed **4/4** varied
inputs (`en`, `16x16`, low-contrast, `3000x80`) for **tiny, small, and
medium** after this change. The five-page public batch gate (also adding dense
`1920x1080`) passed serial-result equivalence for all three models: tiny
**130**, small **126**, medium **129** results. A local one-worker CPU batch
sample measured **0.948x / 1.162x / 1.014x** page-batch throughput for
tiny/small/medium respectively; the mixed dimensions intentionally make this
an evidence point rather than a universal latency claim. Hybrid revalidation
also preserved exact serial results at **1.003x / 1.036x / 0.983x**.

On `en.ppm`, 2 warm-ups and 6 timed CPU runs with 16 workers, the current
native means were **44.193 ms / 84.184 ms / 256.530 ms** for tiny/small/medium.
The matched ONNX Runtime CPU samples were **127.640 ms / 244.623 ms /
513.763 ms**, corresponding to **2.89x / 2.91x / 2.00x** native latency
advantages on this host. Windows scheduling remains variable; these are
repeatable local observations, not a portable capacity guarantee.

The five-size, two-cycle memory soak (`16x16`, `en`, low-contrast, `3000x80`,
`1920x1080`) remained bounded. Its warmed baselines and final deltas were:
tiny **113,876,992 / +3,874,816 bytes**, small **150,462,464 / -2,027,520
bytes**, and medium **385,916,928 / -3,629,056 bytes**. The small movements are
Windows allocator high-water behavior rather than a leak assertion.

### 2026-08-12: batched residual 1x1 projection scheduling

Fused `Conv(1x1) -> Add`, `Conv(1x1) -> Add -> ReLU`, and `Conv(1x1) -> Add
-> Swish` now have NCHW batch kernels. On x86, one work queue covers all
`(crop, output-channel SIMD tile)` tasks: AVX-512 uses the established
four/eight-output tiles and AVX2 uses four-output tiles. ARM/NEON and scalar
builds safely retain the per-crop implementation. The Swish variant completes
the batched fused convolution/add first, then applies the existing vector
Swish once across the whole contiguous NCHW allocation; FP32 reduction and
activation order are unchanged.

`ppocr_kernel_smoke` verifies the new Add, Add+ReLU, and Add+Swish batch
kernels against their single-crop counterparts for both SIMD-tail and large
work cases. It also passed with `PPOCR_DISABLE_AVX512=1`, exercising the AVX2
fallback. `PPOCR_DISABLE_RESIDUAL_POINTWISE_BATCH=1` is available solely as a
per-process A/B escape hatch; it restores the prior per-crop calls.

The four-image decoded-text comparison with ONNX Runtime remained **4/4** for
each tiny, small, and medium model. The five-size public batch-equivalence
gate also matched all serial `Result`s exactly: tiny **130**, small **126**,
medium **129**. One local CPU sample (one detector/recognizer worker,
`rec_batch_size=8`) measured serial/batch times of **909.483/907.159 ms**
(tiny, **1.003x**), **2780.569/2882.623 ms** (small, **0.965x**), and
**6781.126/6750.499 ms** (medium, **1.005x**). Mixed crop widths limit
coalescing, so these are recorded measurements rather than a portability
claim. Hybrid correctness was retained on the same corpus: tiny **1.134 s /
1.132 s** (0.994x), small **3.978 s / 3.515 s** (1.132x), and medium **7.024 s
/ 7.464 s** (0.941x). The hybrid backend still admits each Vulkan segment only
after a complete transfer + dispatch + readback no-slower check; the project
does not claim a fully GPU-resident graph or enable `gpu_only`.

### 2026-08-12: cross-crop CPU depthwise scheduling

The MobileNet depthwise CPU fallback now accepts the leading NCHW batch
directly. It flattens independent `(crop, channel)` jobs into one persistent
worker-pool pass, while each job continues to use the existing runtime-selected
AVX-512, AVX2, NEON, or scalar depthwise kernel. This removes repeated batch
loop scheduling around the dominant depthwise backbone stages without changing
the per-channel FP32 multiply/add order. The Vulkan depthwise route remains
separate and still receives the entire NCHW batch in one dispatch whenever its
complete transfer + dispatch + readback admission measurement is no slower
than CPU.

The extended kernel smoke tests padded/stride-1 `3x3` and padded/stride-2
`5x5` depthwise batches against the established per-image implementation; both
the normal AVX-512 run and `PPOCR_DISABLE_AVX512=1` AVX2 fallback passed. The
four-image decoded-text comparison with ONNX Runtime remained **4/4** for each
tiny, small, and medium model. The five-size batch-result gate likewise
remained exact (tiny **130**, small **126**, medium **129** results).

On the varied five-page corpus with `rec_batch_size=8` and one recognition
worker, this local CPU sample measured serial/batch throughput of **1.096x**
(tiny: 1000.988/913.176 ms), **0.992x** (small: 2855.215/2878.902 ms), and
**1.006x** (medium: 6922.651/6881.270 ms). Crop-width diversity constrains
batch density, so the figures are intentionally recorded as host-specific
measurements. Hybrid preserved serial-result identity at **1.011x**, **1.058x**
and **0.924x** for tiny/small/medium; GPU segments were retained only when the
strict no-slower admission test passed. A renewed five-size/two-cycle memory
soak stayed bounded: tiny baseline **99,737,600**, final **+716,800** bytes;
small baseline **148,520,960**, final **+1,306,624** bytes; medium baseline
**384,311,296**, final **-3,907,584** bytes.

### 2026-08-12: batched ordinary 3x3 convolution scheduler

The group-1 non-pointwise convolution path now supports NCHW batches. For the
common `3x3` stride-1 and stride-2 OCR shapes it flattens crop and established
AVX output-channel tiles into one persistent-pool submission: AVX-512 retains
its four/eight-output tile selection; AVX2 retains four-output stride-1 tiles
and its stride-2 kernel. ARM/NEON and non-3x3 geometries continue through the
existing per-image `Conv2d` path. This changes scheduling only, not the output
channel reduction order. `PPOCR_DISABLE_CONV_BATCH=1` restores the previous
per-crop dispatch for deployment A/B work.

The extended kernel smoke checks stride-1 tail, large AVX-512 tile, and
stride-2 ReLU batches against the prior single-image path; both AVX-512 and
forced AVX2 runs pass. ONNX Runtime decoded-text comparisons again passed
**4/4** for each tiny/small/medium model. On the five-size corpus with one
recognizer worker and `rec_batch_size=8`, the latest CPU serial/batch samples
were tiny **839.965/822.967 ms** (1.021x), small **2602.329/2656.020 ms**
(0.980x), and medium **6507.553/6500.383 ms** (1.001x), all with exact public
batch results. A consecutive A/B sample with `PPOCR_DISABLE_CONV_BATCH=1`
measured **1.023x / 0.966x / 1.005x** on the same mixed-width suite; host noise
and limited same-width coalescing outweigh this small scheduling change there,
so no universal speedup is claimed. Hybrid result equivalence held at
**1.009x / 0.778x / 0.979x**; Vulkan continues to be selected only by its
complete no-slower segment measurement.

### 2026-08-12: batched pointwise HardSwish projection

The remaining fused `1x1 Conv -> HardSwish` CPU path now shares the existing
cross-crop pointwise SIMD scheduler. It performs all NCHW projections through
`PointwiseConvBatch`, then runs one contiguous exact HardSwish pass over the
final allocation, instead of issuing the projection/activation once per crop.
This preserves the Conv FP32 reduction order and the canonical HardSwish
formula; AVX-512, AVX2, NEON, and scalar dispatch remain runtime portable.

Kernel smoke now compares both tail and large batched HardSwish projections to
the established per-crop routine. Normal AVX-512 and forced AVX2 execution
passed. The four-input ONNX Runtime decoded-text gate again passed **4/4** for
tiny, small, and medium, and the varied five-page batch gate remained exactly
identical to serial results (130 / 126 / 129 results). Latest CPU
serial/batch samples on that suite were tiny **901.083/842.421 ms** (1.070x),
small **2515.113/2599.535 ms** (0.968x), and medium
**6177.222/6239.080 ms** (0.990x). Hybrid samples were **0.996x / 1.250x /
0.972x**. These mixed-width measurements are evidence, not portability
claims; each Vulkan segment continues to require its own complete no-slower
transfer/dispatch/readback admission.

### 2026-08-12: batched FPN 2x2 transposed convolution

The detector's `2x2`, stride-two `ConvTranspose` FPN fast path now accepts
NCHW batches. It flattens `(crop, output-channel tile)` into one scheduling
pass: AVX-512 keeps its optional four-output reuse tile (or one-output default)
and AVX2 uses its established channel kernel; ARM/scalar builds retain the
same portable per-image routine. This preserves each output channel's ordered
input-channel accumulation and eliminates repeated batch-loop scheduling.

`ppocr_kernel_smoke` now validates small tail and large square batch cases
against the prior per-image FPN kernel, on both normal AVX-512 and forced AVX2
paths. The tiny/small/medium ONNX Runtime decoded-text checks again passed
**4/4** each. The five-size public batch gate stayed serial-identical at
130/126/129 results. Latest CPU serial/batch observations were tiny
**808.237/821.381 ms** (0.984x), small **2467.217/2557.340 ms** (0.965x), and
medium **6063.649/6071.106 ms** (0.999x). This page mix does not batch detector
shapes and has diverse recognition widths, so it should not be read as a
universal throughput claim. Hybrid exact-result samples were **1.019x / 0.859x
/ 0.974x**. The renewed medium five-size/two-cycle soak remained bounded
(baseline **384,970,752** bytes; final delta **+3,231,744** bytes), and Vulkan
smoke retained strict complete-segment admission.

### 2026-08-12: batched ordinary-convolution Vulkan path

Vulkan hybrid execution now also covers ungrouped, non-dilated ordinary NCHW
convolution (`3x3` and `2x2` detector forms), in addition to the existing
pointwise and depthwise paths.  The new shader maps a real crop batch to its
Y dimension and writes four adjacent output values per invocation where
possible.  Learned `[M,C,KH,KW]` weights and `[M]` bias retain their compact
layouts and are cached for immutable ONNX initializers; an immediately fused
ReLU can complete before the only D2H copy.  Asymmetric-stride, grouped, and
dilated convolutions deliberately retain the established CPU SIMD path.

This remains a *hybrid* primitive, not a GPU-only claim.  Every shape is
first validated against the CPU batched convolution and admitted only when its
complete H2D + dispatch + fence wait + D2H time is no slower.  The new Vulkan
smoke test covers padded stride-one and stride-two `N=3,C=4,M=7,3x3` batches,
including fused ReLU and scalar output tails.  On the local AMD Radeon 780M,
the larger admission sample (`N=4,C=24,M=40,31x29,3x3,ReLU`) measured
**3.53696 ms GPU / 0.83228 ms CPU** after the shared-filter tile optimization,
so the production policy correctly kept
that shape on CPU rather than adding GPU overhead.

After the change, the five-size public batch-equivalence harness preserved
serial output identity for all official model sizes under `PPOCR_BACKEND=hybrid`
with recognizer batch size 8: tiny **130** results (**1188.492 / 1163.981 ms**, 
**1.021x** serial/batch), small **126** (**3425.233 / 3507.859 ms**, **0.976x**),
and medium **129** (**7949.857 / 8198.539 ms**, **0.970x**).  These are
host-specific mixed-size scheduler samples, not a universal GPU-performance
promise.  `gpu_only` remains unavailable until the entire detector and
recognizer graph is device-resident.

### 2026-08-12: final-use BatchNormalization storage reuse

Standard inference `BatchNormalization` nodes now use the same NCHW
channel-affine kernel as fused normalization paths.  When graph liveness
proves that the input has reached its final use, the executor applies the
precomputed `scale / sqrt(variance + epsilon)` and `bias - mean * scale`
directly in that dying allocation, then renames it to the output.  This avoids
one complete activation allocation and copy for each eligible unfused node.
The official model bundle folds most normalisation nodes at load time, so this
extends compatible PP-OCRv6 graph coverage rather than being represented as a
separate current-model speedup.  The new
`BatchNormAffine` primitive is alias-safe and dynamically dispatches through
AVX-512, AVX2, NEON, or scalar `ScaleShift`; it is used in both `Run()` and
the recognizer's real `RunCtcTop1()` path.  `PPOCR_DISABLE_BATCHNORM_INPLACE=1`
is available for deployment A/B checks.

Kernel coverage now includes separate out-of-place and in-place batch-affine
numeric tests, and passed with both default AVX-512 dispatch and forced AVX2.
The post-change C++/ONNX Runtime decoded-text gate passed **4/4** varied
images (`16x16`, normal English, `3000x80` strip, noisy low-contrast) for
both tiny and medium.  A five-size medium two-cycle memory soak remained
bounded at a **457,703,424-byte** post-warm-up baseline; cycles ended at
**-9,195,520** and **+1,744,896 bytes**, consistent with allocator
high-water reuse rather than per-cycle growth.  The same five-size hybrid
tiny batch equivalence test returned all **130** decoded results identically
to serial inference (**1123.880 / 1118.024 ms**, **1.005x** serial/batch).
These are validation measurements, not a claim that every normalisation node
or GPU segment is faster on every host.

### 2026-08-12: batched Vulkan FPN transpose-convolution path

The detector FPN's `ConvTranspose(2x2, stride=2)` projection now has a
dedicated batched Vulkan primitive.  Its no-padding geometry has no overlap
between output tiles, so each output pixel maps directly to one input pixel
and one `[Cin,Cout,2,2]` filter tap; this avoids an expanded intermediate.
The kernel accepts NCHW batches, caches immutable model parameters in the
persistent Vulkan context, and uses one output readback.  The command
dispatcher was also corrected to submit workgroups (256 four-float
invocations per group), rather than accidentally submitting that many times
the required groups.

The hybrid executor now recognizes this exact ONNX form and uses it only
after an end-to-end CPU comparison for the exact batch/shape.  The comparison
includes H2D, dispatch, fence wait, D2H, and FP32 output validation; rejected
shapes continue through the AVX-512/AVX2/NEON/scalar CPU kernel with no output
change.  Vulkan smoke now validates an odd-size `N=3,Cin=5,Cout=7,11x13`
batch against scalar FP32 reference values, including its vector tail.

On the local AMD Radeon 780M, the FPN admission sample
(`N=4,Cin=32,Cout=40,31x29`) measured **2.48820 ms GPU / 0.60534 ms CPU**,
so it was correctly rejected rather than being presented as an OCR speedup.
The post-change five-size tiny hybrid batch-equivalence run remained exactly
serial-identical with **130** decoded results and measured
**1098.189 / 1079.664 ms** serial/batch (**1.017x**).  This is a real
mixed-size batch validation on this host, not a general GPU throughput claim.
`gpu_only` remains unavailable until the full detector and recognizer graphs
can stay device-resident.

### 2026-08-12: FPN transpose shared-filter GPU tile and multi-model batch gate

The Vulkan FPN transpose primitive now selects a shared-filter workgroup form
when its `[Cin,2,2]` row fits the 4,096-float local-memory budget.  A
workgroup stages the non-contiguous ONNX `[Cin,Cout,2,2]` row into contiguous
shared memory once, then computes a 1,024-pixel tile for one output channel.
The staging explicitly preserves the original increasing-input-channel FP32
reduction order.  Wider exported layers retain the generic batch kernel.  The
odd `N=3,Cin=5,Cout=7,11x13` Vulkan numerical test continues to compare every
result with scalar reference values.

On the local Radeon 780M, the complete-transfer FPN admission sample improved
from the preceding **2.48820 ms** to **2.15242 ms GPU**; CPU measured
**0.45748 ms**, so hybrid still correctly selects CPU for that isolated node.
This is intentional: hybrid uses Vulkan when the measured complete segment is
equal or faster, rather than increasing CPU offload at the cost of latency.

The five-size public hybrid batch gate (`16x16`, normal English,
low-contrast, `3000x80`, and dense `1920x1080`; recognizer batch 8, one
recognizer worker) remained serial-identical for all official models: small
**126** results (**2751.279 / 3384.936 ms**, **0.813x**) and medium **129**
results (**7358.610 / 7808.435 ms**, **0.942x**) in this shared-host sample;
tiny's same gate was **130** results (**1098.189 / 1079.664 ms**, **1.017x**).
The C++/ONNX Runtime medium decoded-text gate also passed **4/4** distinct
sizes/aspect ratios (`16x16`, English, low-contrast, wide strip).  The
repository's paired benchmark scripts now accept `-Model tiny|small|medium`,
and `bench/compare_models.ps1` runs all three across the same five-size corpus
for deployment-specific ORT comparison.  Mixed-size batching is not asserted
as a universal speedup; output equivalence and bounded work are the gate.

For a fresh paired single-page sanity sample after this GPU-tile change
(`en.ppm`, two warm-ups, five runs, 16-thread CPU budget, CPUExecutionProvider
for ONNX Runtime), all three official sizes remained faster than ORT:

| Model | C++ hybrid mean | ORT CPU mean | C++ improvement |
|---|---:|---:|---:|
| tiny | 95.490 ms | 128.475 ms | 25.67% |
| small | 157.035 ms | 270.218 ms | 41.89% |
| medium | 263.851 ms | 478.374 ms | 44.84% |

The host is shared and short-run tiny measurements have greater variance; the
table is current measured evidence, not a portable guarantee.  It uses the
same complete detector+recognizer OCR work in both implementations.

### 2026-08-12: terminal CTC guard and scalar-reference coverage

`RunCtcTop1()` already avoids materializing the terminal
`[N,T,6906]` Softmax activation: it keeps final logits, computes only CTC
top-1 indices and probabilities for emitted characters, then releases the
logits at function exit.  This revision narrows the fast-path eligibility to
a verified class-axis terminal Softmax (`axis=-1` or `axis=2`) on the official
`[N,T,V]` exports; a terminal Softmax over any other axis now fails fast rather
than silently applying CTC semantics to the wrong dimension.  The low-level
kernel smoke now compares SIMD-dispatched CTC against a scalar reference for
both a tail-heavy `N=3,T=17,V=31` case and production-width
`N=4,T=41,V=6906`, including blank and repeated-token paths.

Validation after the guard: the expanded CTC kernel smoke passed under both
default AVX-512 dispatch and forced AVX2; C++/ONNX Runtime decoded text passed
**4/4** diverse inputs for tiny and small, while the preceding medium gate was
also **4/4**.  The five-size tiny hybrid batch gate remained serial-identical
at **130** results (**1207.436 / 1297.085 ms**, **0.931x** on this shared
host). A medium five-size/two-cycle memory soak stayed bounded at a
**449,376,256-byte** post-warm-up baseline; cycles ended at **+7,770,112** and
**+5,087,232 bytes**, which is allocator high-water reuse rather than monotonic
per-cycle growth.  Radeon 780M Vulkan smoke remained numerically passing; its
full-transfer policy continued to select GPU only for shapes that measured no
slower than CPU.

### 2026-08-12: hybrid detector batch admission and benchmark controls

`RecognizeBatch()` now permits its existing exact-shape detector NCHW batches
when `Backend::hybrid` is selected as well as on CPU. This does not change
resize semantics or pad differently sized pages: pages are still grouped only
when their independently calculated detector width and height match exactly.
Every Vulkan operator continues to make a separate batch-shape admission
decision that includes H2D, dispatch, fence wait, D2H, and numerical
validation. A rejected operator uses its CPU SIMD batch kernel, so enabling a
larger detector batch never forces a slower partial-GPU path.

`ppocr_bench` now reads `PPOCR_DET_BATCH_SIZE`; `ppocr_batch_smoke` also
reports both effective detector and recognizer batch sizes (and accepts
`PPOCR_REC_WIDTH_BUCKET`). This makes the public batch path reproducible for
deployment A/B work.

On the local Radeon 780M system, eight equal `en.ppm` pages, one page worker,
`rec_batch_size=8`, and `det_batch_size=8` remained exactly serial-equivalent.
One fresh steady-state sample measured CPU **296.587 / 181.402 ms**
(**1.635x**) and hybrid **343.533 / 219.900 ms** (**1.562x**) for serial versus
batched execution. Hybrid with `det_batch_size=1` measured **274.675 /
232.065 ms** (**1.184x**) in a separate run, so this host benefited from
allowing the same-shape detector batch. These are host-specific measurements,
not a claim that every GPU or page mix will have the same ratio.

Regression coverage after the change: normal and AVX2-forced
`ppocr_kernel_smoke` passed; the Vulkan primitive smoke passed on AMD Radeon
780M; and decoded text matched ONNX Runtime for all four varied images in each
of tiny, small, and medium (**4/4 per model**). `gpu_only` remains explicitly
unavailable until the full detector and recognizer graph is device-resident.

### 2026-08-12: fused Vulkan FPN resize/add batch primitive

The FPN's graph-proven `Resize(nearest, 2x) -> Add(lateral)` sequence now has
a Vulkan `NCHW` batch primitive. It uploads the small source map and lateral
residual once, generates each nearest-neighbour value and sums it in one
shader dispatch, then reads only the completed result. The short-lived
expanded resize tensor is therefore never materialized on either CPU or GPU.
Odd dimensions and four-page batch addressing are covered by the Vulkan
smoke test.

Hybrid admission is shape-specific and strict: the measured path includes
source/residual H2D copies, shader dispatch, fence wait, D2H copy, and every
FP32 output comparison against `NearestResize2xAdd`. On this AMD Radeon 780M,
the representative `N=4,C=24,31x29` probe measured **0.47230 ms GPU** versus
**0.06686 ms CPU**, so it is correctly retained on AVX-512 CPU rather than
being forced onto the GPU. This primitive is nevertheless batch-ready for
hardware where the complete path wins.

Post-change validation: default and forced-AVX2 kernel smoke passed; Vulkan
smoke passed; tiny CPU and hybrid decoded-text comparison against ONNX Runtime
passed **4/4** varied inputs; and CPU tiny five-size/two-cycle memory soak
finished at **-139,264** and **-13,721,600 bytes** relative to a
**146,825,216-byte** post-warm-up baseline. A fresh eight-identical-page,
batch-8 smoke sample was exact-equivalent to serial: CPU **303.220 / 181.519
ms (1.670x)** and hybrid **336.976 / 213.733 ms (1.577x)**, serial/batched.
For a single `en.ppm` page with batch sizes one, CPU averaged **37.206 ms**;
hybrid averaged **37.649 ms**, so this shared host's current partial offload
does not claim a whole-pipeline GPU gain. `gpu_only` remains unavailable until
the entire detector and recognizer graphs are device-resident.

### 2026-08-12: recognizer CTC-path lifetime and Vulkan GEMM coverage

The production `RunCtcTop1()` recognizer executor now directly runs fused
`1x1 Conv + shortcut` nodes, matching the ordinary executor rather than
falling through its generic node path. This removes graph-input-vector setup
on the dense residual path and releases the source feature map immediately
after the fused kernel writes its destination. It is shared by tiny, small,
and medium; the three-model CPU/hybrid decoded-text regression against ONNX
Runtime remained **4/4 per model** over tiny, normal English, noisy
low-contrast, and ultra-wide input dimensions.

Vulkan additionally now implements row-major projection GEMM (`[R,K] x
[K,C] + bias`) for recognizer attention and CTC-head-compatible layouts. It
has a numerical smoke for odd `37x29x103` dimensions and a strict full-boundary
admission probe. The Radeon 780M measured the representative `64x128x512`
case at **0.67616 ms GPU** versus **0.15548 ms CPU**, so hybrid correctly keeps
this shape on CPU. The same smoke measured GPU wins for selected batch
depthwise and affine+Swish shapes, which hybrid admits automatically. This
keeps GPU work data-driven and prevents an isolated GPU primitive from
silently increasing complete-OCR latency.

The AVX-512 FPN resize/add kernel was also tightened to load one contiguous
16-float source vector per iteration before two indexed broadcasts; default
and forced-AVX2 kernel smoke pass. On a fresh 16-worker `en.ppm` run (three
warmups, ten timed runs), CPU means were **33.535 / 68.351 / 215.205 ms** and
hybrid means **32.155 / 76.156 / 258.877 ms** for tiny/small/medium. Current
paired ONNX Runtime means from the same session were **176.288 / 229.349 /
572.251 ms**, so CPU remains the deployment recommendation for small/medium
on this UMA GPU while retaining a large native-vs-ORT advantage.

A five-size tiny batch gate (`16x16`, English, low contrast, `3000x80`, dense
`1920x1080`; batch 8) returned all **130** results identically. This mixed
corpus measured **874.808 / 987.314 ms** CPU serial/batched and **1096.915 /
1107.025 ms** hybrid serial/batched, demonstrating that heterogeneous work is
not mislabeled as a throughput win. Its two-cycle memory soak stayed below its
**146,587,648-byte** post-warm-up baseline by **14,483,456** and
**14,401,536 bytes**. `gpu_only` remains explicit-unavailable until complete
detector and recognizer device residency is implemented and verified.

### 2026-08-12: CTC residual GPU admission and true page-batch validation

`RunCtcTop1()` is the production recognizer path, so its fused `1x1 Conv +
shortcut` blocks now use the same batched Vulkan admission path as the generic
executor.  This closes a policy gap where recognizer residual blocks could not
be considered for hybrid execution even when their exact NCHW batch shape had
already been measured.  The decision remains conservative and shape-local:
Vulkan is used only after the full input/residual upload, dispatch, fence wait,
readback, and CPU-SIMD numerical check are no slower than the CPU block.

The change was verified on AMD Radeon 780M with kernel SIMD smoke, Vulkan
primitive smoke, and identical decoded CPU OCR output. A four-page equal-size
tiny batch (`4 x en.ppm`, `det_batch=4`, `rec_batch=4`, one page-scheduler
worker) returned all eight text lines identically and measured:

| Backend | Serial pages | True NCHW page/crop batch | Speedup |
|---|---:|---:|---:|
| CPU | 148.554 ms | 91.244 ms | 1.628x |
| Hybrid | 138.904 ms | 94.528 ms | 1.469x |

This is a same-shape batching result, not a claim for arbitrarily mixed page
sizes. The existing scheduler still groups detector inputs only when their
native PP-OCR resize shape matches and buckets recognizer crops by padded
width, avoiding semantic changes from artificial image padding.

Fresh `en.ppm` end-to-end measurements (16 CPU workers, 3 warmups, 10 timed
runs) show the native engine remains faster than the paired CPU ONNX Runtime
baseline. These are host-specific AMD Radeon 780M / AVX-512-host measurements,
not portable guarantees:

| Model | C++ CPU | C++ hybrid | ORT CPU | CPU latency reduction vs ORT |
|---|---:|---:|---:|---:|
| tiny | 38.215 ms | 37.812 ms | 127.162 ms | 69.9% |
| small | 78.114 ms | 76.542 ms | 220.022 ms | 64.5% |
| medium | 220.072 ms | 240.608 ms | 487.076 ms | 54.8% |

The medium hybrid result is intentionally reported as slower than CPU. Its
individual Vulkan segments may still be admitted only where their complete
boundary beats SIMD; varying transfer/cache pressure on this shared UMA host
can make end-to-end hybrid slower. `gpu_only` still fails explicitly because
the entire PP-OCRv6 graph is not yet Vulkan-resident.

### 2026-08-12: default CPU worker budget tuned for PP-OCR batches

The persistent SIMD worker pool now defaults to **12** workers (still
overrideable with `PPOCR_BENCH_THREADS`). This is a measured scheduling/cache
decision, not a hardware-core-count assumption: on the local AVX-512 host,
the default PP-OCR graph's short detector/recognizer channel tiles saturated
useful memory bandwidth before 16 workers and the extra threads made all three
model sizes slower. The public crop scheduler default is aligned to the same
12-worker budget, avoiding nested page-level oversubscription.

On `en.ppm` with CPU-only execution, three warmups and eight timed runs, the
thread sweep measured the following mean end-to-end latency:

| Model | 8 workers | 12 workers (new default) | 16 workers |
|---|---:|---:|---:|
| tiny | 30.140 ms | 32.457 ms | 35.257 ms |
| small | 70.752 ms | 62.472 ms | 71.507 ms |
| medium | 226.918 ms | 215.528 ms | 226.332 ms |

Tiny stays fastest at eight workers on this host, while the new default gives
the best balanced small/medium result and avoids the former 16-worker
regression. Deployments optimizing only tiny can use
`PPOCR_BENCH_THREADS=8`; larger-server CPUs can continue to tune the existing
override.

Post-change validation covered default and forced-AVX2 kernel smoke, Radeon
780M Vulkan smoke, all three `en.ppm` models, and true NCHW batches. The
five-size corpus (`16x16`, English, low contrast, `3000x80`, dense
`1920x1080`) decoded **130** tiny-model results identically in CPU and hybrid
batch scheduling. Its CPU serial/batched timing was **446.898 / 408.895 ms**;
hybrid was **805.625 / 440.226 ms**. The two-cycle tiny memory soak over that
same five-size corpus ended **10,407,936** and **11,051,008 bytes below** its
128,106,496-byte post-warm-up baseline.

### 2026-08-12: fused Vulkan recognizer GEMM + Swish

The Vulkan recognizer projection primitive now has a `GEMM + Swish` mode for
the graph-fused MLP projection. It accepts the flattened rows of a real crop
batch, caches immutable weights/biases as before, and applies the exact Swish
while writing the GPU result. This removes the former complete CPU-side Swish
traversal after a successful hybrid GEMM dispatch. The shader path is covered
by an odd-tail `41x29x103` numerical smoke case and an end-to-end admission
probe against CPU SIMD `GEMM + Swish`; hybrid still uses it only when the full
upload, dispatch, fence, and readback time is no slower. This is additional
batch-aware GPU coverage, not a claim that the complete PP-OCR graph is yet
device-resident or that `gpu_only` is available.

Validation after the change passed the normal and forced-AVX2 kernel smoke
suites, plus the Radeon 780M Vulkan suite (including the new numerical
GEMM+Swish tail case). On that smoke run the complete-boundary GEMM+Swish
probe measured **0.65962 ms GPU vs 0.15050 ms CPU**, so the dynamic policy
correctly retained CPU for that transfer-bound shape. Tiny-model decoded-text
comparison against ONNX Runtime again passed **4/4** diverse inputs. Four
identical `en.ppm` pages also remained serial-equivalent: CPU measured
**151.283 / 103.396 ms (1.463x)** serial/batched, hybrid
**133.678 / 109.307 ms (1.223x)**. These are local samples; the stricter
per-segment admission rule remains the GPU performance authority.

### 2026-08-12: terminal CTC projection memory experiment

The recognizer executor now contains an optional experimental
`PPOCR_ENABLE_FUSED_TERMINAL_CTC=1` path that computes a terminal
`MatMul(+bias) -> Softmax -> CTC Top-1` directly from one reusable vocabulary
row buffer. It eliminates the transient `[N,T,V]` logits allocation and
preserves the dense projection's K-order plus exact selected-class Softmax
probability. It is deliberately opt-in: on the local AVX-512 CPU the scalar
vocabulary-row implementation is not a stable latency win for every official
model, although it is useful for memory-constrained deployments and passed
the text regression on the exercised corpus. The default continues to use the
proven SIMD GEMM plus vectorized CTC scan until a native SIMD fused terminal
kernel demonstrates a three-model speedup.

The row-buffer path is now covered in the SIMD kernel smoke with both a
small odd-tail projection and a `4x41x6906` recognizer-sized projection.
The portable regression corpus was also expanded beyond the original four
shapes: blank `800x600`, English and CJK single lines, a mixed three-line
receipt, and two-line CJK all decode with ONNX Runtime text parity for tiny,
small, and medium (**15/15 per model**). Deliberately adversarial rotation and
extreme portrait-strip images are retained as diagnostic samples, but are not
used as decoded-text parity gates because the two independently implemented
detector postprocessors can select different unstable low-confidence boxes.

### 2026-08-12: cooperative Vulkan GEMM tile and larger page batches

The Vulkan recognizer GEMM now has a cooperative **8-row 鑴?128-column** FP32
tile for ordinary (non-Swish) projections. Each workgroup stages a 16-wide K
slice of A and B, reusing B across eight rows rather than rereading it per
output element. The kernel keeps K tiles in ascending order, retains the
existing scalar-tail path, and is selected only after the existing exact
CPU-versus-full-boundary admission check. `gpu_only` remains unavailable: this
is a batch-aware primitive within the hybrid executor, not a device-resident
PP-OCR graph.

The new path is covered by Vulkan numerical smoke with a tail-heavy
`13鑴?7鑴?73` GEMM (rows, K, columns), in addition to the existing odd-tail and
GEMM+Swish cases. Normal SIMD smoke and Vulkan smoke passed on the local
AVX-512 / Radeon 780M host. Hybrid decoded-text parity against ONNX Runtime
again passed 4/4 diverse inputs for both tiny and medium (`16x16`, English,
low-contrast, and `3000鑴?0`).

For an eight-page same-shape `en.ppm` batch with `det_batch_size=8`,
`rec_batch_size=8`, and one recognizer scheduler worker, the post-change
hybrid measurements were tiny **304.188 / 225.182 ms (1.351x)** and medium
**2029.114 / 2019.849 ms (1.005x)** serial/batched. A four-shape hybrid batch
gate (`16x16`, English, low-contrast, `3000鑴?0`; batch sizes 4) remained
serial-identical for all models: tiny **186.390 / 176.957 ms (1.053x)**, small
**379.836 / 371.670 ms (1.022x)**, medium **1209.497 / 1215.288 ms (0.995x)**.
These are one host's steady-state samples, not a universal GPU claim; the
per-shape GPU admission continues to reject transfer-bound work.

After the latest pass, default CPU/hybrid batch regression again produced
serial-identical results across four different page shapes (`16x16`, English,
low-contrast, and `3000x80`). With bounded `det_batch_size=4` and
`rec_batch_size=4`, the local CPU/hybrid batch timings were tiny
**176.722/185.384** and **170.369/165.231 ms**; small **343.559/301.114** and
**374.170/293.868 ms**; medium **1317.190/1291.101** and
**1163.054/1122.104 ms**. A medium two-cycle varied-size memory soak started
at **495,898,624 bytes** after warm-up and ended **9,752,576** then
**10,174,464 bytes below** that baseline. These are host-local measurements,
not a general ONNX Runtime comparison claim.

### 2026-08-12: optional PP-OCRv6 vector GELU validation

The existing optional recognizer vector tanh-GELU path
(`PPOCR_APPROX_GELU=1`) was revalidated on AVX-512/AVX2; NEON/scalar retains
its portable fallback. This removes scalar `erf` from the MLP activation hot
path. The exact-Erf ONNX path remains the default, and the model-specific
vector path was checked against ONNX Runtime decoded text rather than being
assumed numerically interchangeable.

With 12 worker threads, three warmups and ten `en.ppm` runs, the scalar-Erf
baseline measured **29.097 / 60.268 / 202.201 ms** for tiny/small/medium; the
optional vector path measured **23.577 / 53.054 / 191.974 ms** in the same
session (**19.0% / 12.0% / 5.1%** lower mean latency). A longer shared-host
30-run medium sample was noisier, so the 10-run paired sweep is recorded as
an optimization signal, not a portable latency guarantee.

The decoded-text parity gate passed **4/4** diverse dimensions against ONNX
Runtime for each official model size (tiny, small, medium): `16x16`, English,
low-contrast, and `3000x80` ultra-wide. CPU and hybrid also produced identical
Chinese tiny-model text for the CJK sample. The kernel smoke still passes with
normal AVX-512 dispatch and forced AVX2 fallback.

### 2026-08-12: AVX-512 batch-GEMM dispatch refinement

The AVX-512 dense-GEMM path now selects its four-row by 64-column micro-kernel
per projection shape instead of applying its extra register pressure to every
recognizer tier. It stays enabled only for `K >= 384` (and can be disabled for
A/B with `PPOCR_DISABLE_AVX512_GEMM4ROW=1`); AVX-512 閳?AVX2 閳?scalar/NEON
runtime dispatch remains unchanged for all x86 and Arm targets. This keeps
large medium-model projections eligible to reuse each immutable B vector over
four output rows, while tiny/small use the lighter proven path.

Fresh `en.ppm` CPU measurements (three warm-ups, ten runs, 12 kernel workers)
were tiny **29.699 ms**, small **59.577 ms**, and medium **196.789 ms**. The
previous all-shape four-row experiment measured small **82.637 ms** and medium
**267.150 ms** in the same benchmark configuration, so the new admission
guard is retained. A dense `1920x1080` medium page (122 boxes,
`rec_batch_size=4`, `rec_parallelism=4`, one warm-up, three runs) measured
**4,496.430 ms** with the wide tile enabled and **4,426.272 ms** with it
disabled; therefore the dense-page default now avoids that tile whenever its
projection depth is below the measured `K=384` cutoff. These are
host-local figures, not an ONNX Runtime claim.

The normal AVX-512 kernel smoke passed after the change, as did the complete
Radeon 780M Vulkan smoke suite. The latter again verified all GPU batch
primitives numerically and dynamically admitted only full-boundary winners:
affine+Swish **0.17342 vs 0.34950 ms**, depthwise **0.34814 vs 0.47170 ms**,
depthwise+Swish **0.35624 vs 0.60330 ms**, and depthwise+HardSwish
**0.34528 vs 0.41760 ms** (GPU vs CPU). Transfer-bound GEMM and other
primitives stayed on CPU; `gpu_only` is still explicitly unavailable until
the complete PP-OCR graph is device-resident.

### 2026-08-12: diverse-size regression and preprocessing arithmetic reduction

The preprocessing resize path now folds PP-OCRv6's fixed input normalization
into precomputed per-channel affine coefficients. It keeps the existing
bilinear rounding, RGB-to-BGR layout, detector mean/std, and recognizer
`[-1,1]` arithmetic, while removing three per-pixel floating-point divisions
from detector and crop preparation. This applies to serial OCR and true
`[N,3,H,W]` detector/recognizer batches, without adding a temporary image
buffer or changing the x86 AVX-512/AVX2/scalar/NEON runtime dispatch.

The post-change ONNX Runtime decoded-text gate passed **8/8** diverse PPM
sizes for every official model (**24/24 total**): `16x16`, English, noisy
low-contrast, `3000x80`, English/CJK single lines, mixed text, and CJK. A
four-page CPU batch smoke across the smallest, normal, low-contrast, and
ultra-wide samples remained byte-for-byte result equivalent to serial OCR
and measured **190.614 / 176.096 ms (1.082x)** serial/batched for tiny.
This explicitly exercises multiple aspect ratios rather than using a
same-page-only batch claim.

A fresh `en.ppm` three-warmup/ten-run snapshot (12 kernel workers) measured
CPU tiny **35.073 ms**, small **72.582 ms**, medium **244.095 ms**; hybrid
measured **34.887 / 72.036 / 244.847 ms**. The hybrid implementation still
selects Vulkan only after per-shape numeric validation of the complete
upload, dispatch, fence, and readback boundary; therefore it may reduce CPU
work without claiming a universal latency gain on UMA hardware. `gpu_only`
remains an explicit failure until all PP-OCRv6 operators are device-resident.

The expanded five-size memory soak (`16x16`, English, low-contrast,
`3000x80`, dense `1920x1080`) was also run twice per model. It showed no
unbounded retained activation frontier: tiny ended **14.63 MiB below** its
146,534,400-byte warm baseline; small and medium ended **8.66 MiB** and
**11.69 MiB** above their 178,892,800-byte and 425,119,744-byte baselines.
Those modest Windows allocator deltas are reported rather than hidden; a
future allocator-pool change must improve this measured dense-page result
without weakening the 24-case text-parity gate.

### 2026-08-12: dense recognizer GEMM tile admission refinement

The medium dense-page operator profile identified transformer dense projection
as the largest recurring recognizer cost (`FusedMatMulBias` was **28.685 ms**
for one profiled crop batch), ahead of any scalar postprocessing. The AVX-512
four-row GEMM micro-kernel therefore now requires both `K >= 384` **and**
`N >= 384`. This keeps it on the PP-OCRv6 medium MLP expansion
`192x768` while retaining the lower-register-pressure two-row implementation
for the paired `768x192` projection. The runtime CPU hierarchy is unchanged:
AVX-512 is selected only after runtime CPUID/XCR0 checks, then AVX2/FMA,
then portable scalar/NEON.

On the dense `1920x1080` medium UI (122 boxes, `rec_batch_size=4`,
`rec_parallelism=8`, eight kernel workers, one warm-up/three runs), the
refined default measured **4,328.474 ms**; disabling the four-row tile for
A/B gave **5,303.048 ms** (with a 6,549 ms shared-host tail). On `en.ppm`,
the same model measured **210.125 ms** versus **210.604 ms** disabled across
three warmups/ten runs. The normal AVX-512 kernel smoke, all three models'
eight-shape ONNX Runtime decoded-text gates (**24/24**), the Radeon 780M
Vulkan batch smoke, and a varied-shape hybrid batch identity check all passed.
The hybrid check continues to admit Vulkan only after complete-boundary
validation; in this sample it selected affine+Swish and depthwise variants
where GPU was faster, while keeping transfer-bound work on CPU.

### 2026-08-12: GPU-aware NCHW batch benchmark controls

`ppocr_batch_smoke` now supports repeatable steady-state measurements through
`PPOCR_BATCH_WARMUP` and `PPOCR_BATCH_RUNS`. It still verifies that every
batched page result (text, confidence, axis-aligned box, and four-point box)
matches serial inference exactly, then reports mean and median serial/batch
times. This exercises the existing real `[N,3,H,W]` detector batches and
cross-page recognizer crop batches; it is not a scheduler-only benchmark.

On the local Radeon 780M / AVX-512 host, four identical `en.ppm` tiny pages
with `det_batch_size=4`, `rec_batch_size=4`, one outer worker, two warm-ups,
and five timed runs produced eight serial-identical results. CPU measured
**192.813 / 137.304 ms** serial/batched (**1.404x**); hybrid measured
**175.306 / 125.126 ms** (**1.401x**). The hybrid result remains governed by
per-shape full H2D + dispatch + fence + D2H validation, so a GPU segment is
used only when it measured no slower than SIMD CPU for that exact batch shape.

The same hybrid configuration, with two warm-ups and three timed runs, gave
small **330.788 / 308.714 ms** (**1.072x**) and medium **991.291 / 1061.823
ms** (**0.934x**) serial/batched. The medium result is intentionally retained:
on this shared UMA host, combining page/crop work can lose to cache pressure
and transfer-bound segments. Batch support therefore does not claim a
universal GPU gain, and `gpu_only` remains unavailable until the full graph is
device-resident.

The latest run also reconfirmed dynamic high-ISA dispatch. With 12 kernel
workers and three warm-ups/six `en.ppm` runs, shape-gated AVX-512 pointwise
eight-output tiling measured tiny/small/medium **35.979 / 76.015 / 252.816
ms**. Disabling only that tile via `PPOCR_DISABLE_AVX512_POINTWISE8=1`
measured **36.286 / 81.219 / 268.576 ms**. The tile remains limited to
`C_in >= 192`, `C_out >= 192`, and plane >= 256; AVX-512 selection is runtime
CPUID/XCR0-gated, followed by AVX2 and portable scalar/NEON fallbacks.

### 2026-08-12: repeatable batch, Vulkan, memory, and model-size regression

This pass kept the production CTC executor's terminal-logit memory mode
opt-in (`PPOCR_ENABLE_FUSED_TERMINAL_CTC=1`): it replaces a full terminal
`[N,T,V]` logits activation with a per-sequence row workspace, but remains
off by default because the regular SIMD GEMM path is faster on this host.
This preserves the lower peak-memory option without making an unmeasured
throughput trade-off the default.

The complete CPU-side five-size private-memory soak was rerun with 12 kernel
workers over `16x16`, English, low-contrast, ultra-wide `3000x80`, and dense
`1920x1080` inputs. After all shapes were warmed, tiny finished cycle two
**10.09 MiB below** its **126,599,168-byte** baseline. Small and medium one
post-warm cycles reported **+4.07 MiB** from **178,348,032 bytes** and
**+10.47 MiB** from **425,025,536 bytes** respectively. These allocator
deltas are reported as observed rather than treated as a leak verdict.

The decoded-text ONNX Runtime regression passed the full eight-shape corpus
for tiny and small (**8/8 each**), and the cross-model boundary subset
(`16x16`, English, low-contrast, `3000x80`) passed medium (**4/4**). The
Radeon 780M Vulkan primitive suite also passed numerical validation. In its
latest sample hybrid admitted affine+Swish (**0.18734 vs 0.37020 ms** GPU/CPU),
depthwise+Swish (**0.33626 vs 0.61096 ms**), and depthwise+HardSwish
(**0.31810 vs 0.34620 ms**); transfer-bound projection, GEMM, ordinary Conv,
and resize work stayed on CPU automatically.

Finally, a paired six-run medium English benchmark with two warm-ups and 12
kernel workers measured pure C++ CPU **259.037 ms** versus ONNX Runtime CPU
**297.913 ms** (**13.1% lower latency**, **3.860 vs 3.357 FPS**). The ORT
sample had shared-host variation, so this is a current reproducible snapshot,
not a universal multiplier.

### 2026-08-12: optional large-map SIMD GELU admission

`PPOCR_APPROX_GELU=1` now applies the PP-OCR-specific vector GELU
approximation only to tensors with at least 65,536 values. This avoids the
short, decode-sensitive boundary activations while allowing the AVX-512/AVX2
path to amortize its approximation on genuine large feature maps. Exact
Erf-GELU remains the default, and Arm/NEON remains on the portable exact path.

With 12 kernel workers, three warm-ups, and ten English-page runs, the gated
opt-in mode measured tiny/small/medium **27.295 / 53.594 / 204.034 ms**,
versus exact-default **28.491 / 61.181 / 214.164 ms** in the paired session:
**4.2% / 12.4% / 4.7%** lower latency. The opt-in decoded-text gate passed
tiny **8/8**, small **8/8**, and medium **4/4** across the same diverse
dimensions used above. A mixed-size hybrid page/crop batch also remained
serial-identical (four pages, seven results); it measured **161.524 /
164.530 ms** serial/batched on this heterogeneous mix, so no universal batch
speedup is claimed for unlike page shapes.

### 2026-08-12: Vulkan batch revalidation

After rebuilding the Vulkan batch path, the Radeon 780M primitive smoke test
again passed numerical validation. The complete-boundary policy kept the
transfer-bound binary, pointwise, depthwise, GEMM, ordinary-convolution and
resize probes on CPU; affine+Swish was the only selected GPU primitive in the
latest pass (**0.42584 vs 0.43290 ms** GPU/CPU). These are per-primitive
measurements including H2D, dispatch, fence and D2H, not a claim that the
whole OCR graph is GPU-resident.

On four same-shape `en.ppm` pages with hybrid selected, one page worker,
`det_batch_size=4`, `rec_batch_size=4`, one warm-up and three timed runs,
the serial/batched comparison was **148.272 / 115.169 ms** (**1.287x**).
The result check remained exact at the public OCR result level (text, box and
confidence). The scheduler sends the real `N=4` tensor to both the CPU SIMD
and Vulkan primitive batch dimensions; heterogeneous inputs remain governed
by the existing width/shape buckets instead of being padded arbitrarily.

### 2026-08-12: GPU batch-path cleanup and recheck

The unvalidated experimental pointwise-to-depthwise Vulkan chain was removed.
It had never been reachable from the hybrid executor and had previously failed
its numerical validation, so retaining it would have overstated device-resident
graph support. The production Vulkan shader now contains only smoke-tested
primitive modes. `gpu_only` therefore continues to reject execution explicitly;
the implementation must not silently fall back to CPU until the complete
PP-OCRv6 graph, DB post-processing, and CTC decode can remain on the device.

After the cleanup, the Radeon 780M Vulkan primitive smoke and AVX-512 kernel
smoke both passed. The same four-page tiny English batch (`det_batch_size=4`,
`rec_batch_size=4`, one outer worker, one warm-up, three timed runs) remained
serial-identical: CPU was **140.573 / 100.661 ms** serial/batched (**1.396x**),
and hybrid was **135.904 / 99.425 ms** (**1.367x**). A hybrid ONNX Runtime
text gate also passed **4/4** diverse tiny inputs (16x16, English,
low-contrast, and 3000x80 ultra-wide). These figures exercise the public
real-N detector and recognition batches; they do not claim a complete
GPU-resident inference path.

### 2026-08-12: fused approximate BatchNorm-GELU

When the existing opt-in `PPOCR_APPROX_GELU=1` mode is enabled, a large
fused BatchNorm-GELU group now performs its channel affine and AVX-512/AVX2
Pad鑼?GELU approximation in one vector pass. This removes the previous
per-plane affine write/read round trip. The normal exact-Erf path is unchanged;
non-x86/NEON builds also retain its exact behavior. Set
`PPOCR_DISABLE_FUSED_BN_GELU=1` only to reproduce the former two-pass
experimental implementation for A/B measurement.

On the local AVX-512 host with 12 worker threads, two warm-ups and six English
runs, the fused mode measured tiny **27.106 ms**, small **51.901 ms**, and
medium **225.597 ms**. The former two-pass opt-in path measured **27.241 / 
62.450 / 232.207 ms** respectively, a material small-model reduction and a
positive medium improvement in that paired session. Decoded-text regression
against ONNX Runtime passed the eight-size corpus for tiny, small, and medium
(**24/24** total): 16x16, English, low-contrast, ultra-wide, English/CJK
single-line, mixed text, and CJK. The broad requirement to outperform ONNX
Runtime on every model and workload remains a continuing target, not a claim
from this one host-local optimization sample.

### 2026-08-12: batch-invariant GELU admission correction

The optional SIMD GELU admission was tightened so batching cannot change a
page's numerical execution path. Generic GELU is now always exact, because it
has no NCHW shape information and a total-element threshold could make a
singleton crop use Erf while the same crop in a batch used the approximation.
The model-aware fused BatchNorm+GELU path continues to use its AVX-512/AVX2
one-pass approximation only when each NCHW plane is at least 65,536 elements;
this predicate is invariant to the leading batch dimension. Exact GELU remains
the default and ARM/NEON retains the exact fallback.

After the correction, kernel and Radeon 780M Vulkan smoke both passed. The
hybrid decoded-text ONNX Runtime gate passed **4/4** tiny diverse inputs
(16x16, English, low contrast, and 3000x80), with `PPOCR_APPROX_GELU=1`.
The public four-page `en.ppm` batch gate also restored exact serial/batch
results at `det_batch_size=4`, `rec_batch_size=4`, and one recognition worker:
**169.174 / 123.454 ms (1.370x)** for its three-run hybrid sample. A longer
ten-run local snapshot measured CPU **165.446 / 118.013 ms (1.402x)** and
hybrid **210.649 / 142.646 ms (1.477x)**. These shared-host figures are batch
throughput evidence, not an end-to-end GPU-only performance claim; hybrid
still admits only individually validated full-boundary Vulkan segments.

### 2026-08-12: MatMul write-only activation allocation

The constant-weight transformer MatMul path now allocates its destination with
`resize()` rather than value-initializing it to zero. `kernels::Gemm()` writes
every output element from its bias/zero accumulator, so the previous clear was
dead memory traffic before the SIMD or Vulkan attempt. This reduces activation
write bandwidth and allocator work for the recognizer's tall batch-folded
projections without changing the CPU, hybrid, or ARM/NEON fallback semantics.

Post-change kernel and Vulkan smoke passed, including the Radeon 780M
full-boundary admission checks. Hybrid decoded-text parity against ONNX Runtime
passed the eight-size corpus for every supported model (**24/24**): 16x16,
normal English, low contrast, 3000x80, English/CJK lines, mixed text, and CJK.
A separate eight-image mixed-aspect tiny hybrid batch check returned all 14
serial-identical OCR results at **264.126 / 256.614 ms (1.029x)**; this is a
diversity/correctness gate and is intentionally not represented as a universal
throughput claim.

For a current 12-worker, three-warmup/twenty-run `en.ppm` CPU comparison on
this shared host, the measured C++ / ONNX Runtime means were tiny **34.049 /
85.073 ms** (60.0% lower C++ latency), small **75.880 / 106.510 ms** (28.8%
lower), and medium **249.121 / 210.750 ms** (18.2% higher). The medium sample
does not satisfy the project performance target and is recorded explicitly:
the remaining work is medium recognizer/detector optimization and longer
device-resident Vulkan segments, not a claim that this pass has already made
all models faster than ONNX Runtime.

### 2026-08-12: multi-adapter Vulkan selection for batched hybrid execution

The Vulkan backend now evaluates all compute-capable adapters instead of
blindly selecting Vulkan's first enumerated queue. Its default selection is
the integrated/UMA adapter, which is the appropriate default for the current
per-segment H2D/dispatch/D2H hybrid design; set
`PPOCR_VULKAN_PREFER_DISCRETE=1` to prefer a discrete accelerator. Deployment
integration can pin an adapter with `PPOCR_VULKAN_DEVICE_INDEX=N` or select it
by substring through `PPOCR_VULKAN_DEVICE_NAME=...`. CPU Vulkan drivers are
not chosen automatically, but may still be requested by index for diagnostic
purposes. Every selected operator remains gated by its exact shape/batch,
full-boundary no-slower-than-CPU measurement.

After this change, the Radeon 780M Vulkan smoke suite and AVX-512 kernel smoke
both passed. A four-page same-shape tiny batch (`det_batch_size=4`,
`rec_batch_size=4`, hybrid, one page worker, two warmups/five timed runs)
returned all eight public OCR results identically. The shared Windows host
showed unstable timing in the immediate post-build sample (**170.452 ms serial
/ 219.688 ms batch; batch median 119.245 ms**), so no new throughput claim is
made from that run. It nevertheless validates real `[N,3,H,W]` detector and
crop batches on the selected GPU path; it does not represent a GPU-only claim
because the complete graph, DB post-processing, and CTC decode are still not
device-resident.

### 2026-08-12: GPU batch front-end parallelism

The public batched API now exposes `Options::batch_preprocess_parallelism`
(default **4**), with the benchmark override
`PPOCR_BATCH_PREPROCESS_PARALLELISM`. For a same-shape detector batch, each
page's resize/BGR-normalize write owns a separate contiguous NCHW plane and is
prepared concurrently before the one batched CPU/Vulkan graph submission.
The budget is divided across concurrent dynamic-shape detector batches, so
raising `image_batch_parallelism` does not cause unbounded nested-thread
oversubscription. It is CPU front-end work that enables GPU/SIMD batching; it
does not change the strict Vulkan transfer/admission policy or imply a
GPU-only graph.

On the local Radeon 780M / AVX-512 host with the tiny model, four identical
`en.ppm` pages, `det_batch_size=4`, `rec_batch_size=4`, one page worker,
`PPOCR_APPROX_GELU=1`, three warm-ups, and ten timed hybrid runs, the
serial-identical public result check passed in both configurations. Serial
preprocess (`PPOCR_BATCH_PREPROCESS_PARALLELISM=1`) measured **121.887 ms**
batched mean (**1.479x** against **180.328 ms** serial); the default four-way
front end measured **119.919 ms** (**1.498x** against **179.646 ms** serial).
This is a **1.6%** improvement in that controlled local batch sample. Timing
on the shared Windows host remains variable, so the README records it as a
reproducible configuration-specific result rather than a universal GPU gain.

### 2026-08-12: experimental Vulkan tiled GEMM+Swish batch kernel

The Vulkan shader now includes an 8-row x 128-column cooperative GEMM tile
with Swish fused into its final writeback (mode 25). It shares a 16x128 weight
tile across eight flattened recognition rows, so it is designed for real
same-width crop batches rather than per-crop dispatches. Its FP32 K reduction
remains ascending, and the shader stays behind the existing numerical smoke
and complete H2D + dispatch + fence + D2H admission checks. Enable it only for
driver-specific A/B work with `PPOCR_ENABLE_VULKAN_GEMM_SWISH_TILE=1`; the
plain tiled GEMM path remains dynamically selected as before.

On the local Radeon 780M, AVX-512 host, the experimental tile passed kernel
and Vulkan primitive smoke plus the hybrid ONNX Runtime decoded-text gates for
tiny and medium (4/4 diverse inputs each). A quick medium four-page A/B saw
the tile at **1004.444 ms** batched mean and the default at **1041.713 ms**;
the longer shared-host samples varied enough that neither result is promoted
to the default. The default public medium batch remained result-identical and
measured **1178.757 / 1061.702 ms** serial/batched (**1.110x**) with two
warm-ups and six timed runs. This is an implementation and correctness
milestone, not a claim of a universal GPU speedup; hybrid continues to select
GPU only for primitives that win their exact full-boundary admission
measurement.

### 2026-08-12: recognition-batch front-end parallelism

`batch_preprocess_parallelism` now also prepares same-width recognition crops
in parallel. Each resize owns one already-zeroed, contiguous `[N,3,48,W]`
plane; output order, natural resize width, right padding, and normalization
arithmetic are unchanged. The front-end budget is divided by concurrent
recognizer batches, avoiding nested oversubscription when
`rec_parallelism > 1`.

On the local Radeon 780M / AVX-512 host, four same-shape tiny English pages
with hybrid mode, `det_batch_size=4`, `rec_batch_size=4`, one recognizer
worker, two warm-ups and eight timed runs were public-result identical. A
serial front end measured **112.023 ms** batch mean; four-way front-end work
measured **105.894 ms**, a **5.5%** batch-latency reduction. The corresponding
serial/batch speedups were **1.523x / 1.577x**. A five-run small-model check
remained result-identical too (**314.876 / 313.329 ms** batch at one/four-way
preprocess); it is kept as evidence that the gain is input/model dependent,
not as a universal throughput claim. Kernel and Vulkan primitive smoke passed
after the change; decoded-text ONNX Runtime checks also passed the small
model's 16x16, English, low-contrast and ultra-wide inputs. Very dense/tall
synthetic pages remain a deliberately stricter diagnostic corpus: their
near-threshold DB boxes can differ from ORT by one pixel and therefore must
not be used as a text-parity gate without a geometric tolerance.

### 2026-08-12: GPU-ready single-page crop batching

The same crop-plane parallelism is now used by both public entry points:
`RecognizeBatch()` and ordinary `Recognize()`. A dense single page can contain
several same-width recognition crops, so this closes a scheduling gap where
only multi-page batches could prepare their real `[N,3,48,W]` input planes in
parallel. The budget is `Options::batch_preprocess_parallelism` (or
`PPOCR_BATCH_PREPROCESS_PARALLELISM`) and is divided by active recognizer
batches, which prevents nested CPU oversubscription. Each worker owns a
pre-zeroed NCHW plane; width padding and every pixel/normalization calculation
remain unchanged.

This benefits the existing CPU SIMD and Vulkan hybrid paths equally. Vulkan
continues to receive a genuine NCHW batch, with its batch dimension mapped to
the shader dispatch Y dimension. It still offloads only an operator after its
same-shape, full H2D + dispatch + fence + D2H numerical/performance admission
test is no slower than the CPU kernel; `gpu_only` remains unavailable because
the complete PP-OCRv6 graph, DB postprocessing, and CTC decode are not
device-resident.

Post-change verification passed the AVX-512 kernel smoke, Vulkan primitive
smoke, four-page public result identity, and a medium varied-shape hybrid
memory soak. On the local Radeon 780M / AVX-512 host, four equal `en.ppm`
tiny pages in hybrid mode (`det_batch_size=4`, `rec_batch_size=4`, one page
and recognizer worker, two warm-ups, six timed runs) measured **117.702 ms**
with serial front-end work and **110.992 ms** with a four-way front end: a
**5.7%** batch-latency improvement (serial/batch speedup **1.420x ->
1.498x**). A short medium CPU batch check also preserved all eight public OCR
results and measured **1118.404 / 1021.506 ms** serial/batched (**1.095x**).
These shared-host figures are configuration-specific benchmarks, not a claim
that every model or GPU segment is faster.

### 2026-08-12: experimental AVX-512 wide-context tile

Medium PP-OCRv6 detector context blocks include padded 5x5 and 7x7
convolutions. An experimental AVX-512 eight-output tile now shares each
16-float input load across eight output channels, while keeping the existing
four-output implementation for a non-multiple-of-eight tail. It preserves
each output's increasing input/channel/kernel FMA order and has no effect on
AVX2, ARM/NEON, or scalar dispatch. Enable it only for deployment A/B work
with `PPOCR_ENABLE_AVX512_CONV_WIDE8=1`; the established four-channel tile
remains the default until it wins reliably across target CPUs.

Both default and opt-in paths passed the kernel smoke, including padded 5x5
and 7x7 eight-channel bodies plus nine-channel tails, and the full Radeon
780M Vulkan primitive suite. On this shared AVX-512 host, a medium `en.ppm`
CPU sample (two warm-ups/eight runs, twelve workers) measured **223.342 ms**
default and **221.962 ms** opt-in (0.6% lower mean). A 122-crop
`ui_dense_1920x1080` sample did not reproduce that gain (**4053.814 vs
4075.517 ms**, two warm-ups/five runs), so this remains deliberately opt-in
rather than a broad performance claim. Hybrid Vulkan admission remains
unchanged: every GPU primitive is still selected only when its validated full
H2D + dispatch + fence + D2H boundary is no slower than CPU.

The same post-change CPU 12-worker `en.ppm` snapshot (two warm-ups/ten runs)
was tiny **36.902 ms**, small **72.035 ms**, and medium **272.094 ms**.
Hybrid runs (two warm-ups/six runs) were **37.502**, **72.662**, and
**255.454 ms** respectively. The medium hybrid improvement is useful on this
UMA GPU, while CPU remains slightly better for tiny/small in this run; the
per-primitive admission rule intentionally preserves that outcome instead of
forcing Vulkan. Medium decoded-text parity against ONNX Runtime also remained
**4/4** across 16x16, English, low-contrast, and 3000x80 inputs with the
experimental wide tile enabled. These measurements do not change the prior,
more representative ONNX Runtime comparison: medium has not yet demonstrated
a consistent end-to-end lead over ORT and remains the main optimization target.

### 2026-08-12: fused GPU ordinary-Conv activation batches

The hybrid Vulkan ordinary NCHW convolution primitive now also accepts
graph-fused **Swish**, Sigmoid, and HardSwish writebacks. Non-pointwise
detector/recognizer Conv activation chains therefore use one batched GPU
submission, with no GPU-to-CPU readback merely to apply the nonlinear pass.
The CPU SIMD fallback is unchanged. Admission remains exact per
`(N,C,M,H,W,kernel,stride,padding,activation)` and includes H2D, dispatch,
fence, D2H, and output validation against CPU Conv plus the same activation.
If this complete boundary loses, hybrid keeps CPU; `gpu_only` remains
unavailable.

`ppocr_vulkan_smoke` now prints an independent `Conv+Swish` batch admission
measurement next to its existing `Conv+ReLU` result. This is primitive-level
evidence only; an end-to-end OCR GPU claim requires the corresponding shapes
to pass their dynamic admission measurements.

Post-change AVX-512 kernel smoke and Radeon 780M Vulkan smoke both passed.
The new ordinary `N=4,C=24,M=40,31x29,3x3` Swish probe measured **3.58512 ms
GPU vs 1.24636 ms CPU**, so hybrid correctly retained CPU for that segment on
this UMA device. A hybrid four-page exact-shape batch regression also remained
result-identical: tiny measured **109.844 / 84.810 ms** serial/batched
(**1.295x**), and medium **920.012 / 836.339 ms** (**1.100x**) with two
warm-ups and three timed runs, `det_batch_size=4`, `rec_batch_size=4`, and
four-way preprocessing. Medium decoded-text ONNX Runtime parity passed **4/4**
for 16x16, English, low-contrast, and ultra-wide inputs. These are controlled
batch measurements on a shared host, not a GPU-only or universal throughput
claim.

### 2026-08-12: liveness-aware in-place Concat

Both graph executors now reuse the first dynamic input of a final-use ONNX
`Concat` as the output storage. The general implementation handles every
legal axis: after growing the vector, it moves the first input's outer blocks
backwards (overlap-safe) and copies only the additional branch spans. This is
especially relevant to the medium detector's channel-axis FPN assembly, where
operator profiling previously exposed `Concat` as a measurable cost. It avoids
one complete output allocation and the copy of the first branch while keeping
the generic path for shared/aliased/non-final-use inputs. `RunCtcTop1()` has
the same rule, so public recognizer batches do not diverge from `Run()`.

Verification after the change passed AVX-512 kernel smoke, Radeon 780M Vulkan
smoke, and medium CPU decoded-text ONNX Runtime parity on 16x16, English,
low-contrast, and 3000x80 inputs (**4/4**). A medium hybrid seven-size memory
soak (16x16, English, low contrast, ultra-wide, ultra-tall, dense 1920x1080,
and 2560x1440 screenshot) finished one cycle at **+14,647,296 bytes** over a
453,398,528-byte post-warmup private-byte baseline. This is allocator
high-water behavior, not a leak assertion. The immediate six-run medium
`en.ppm` CPU sample measured **234.826 ms**; hybrid measured **258.733 ms** on
this shared UMA host, so the dynamic per-segment rule correctly does not claim
a blanket GPU gain.

### 2026-08-12: hybrid GPU batching and queue-aware scheduling

The Vulkan path continues to accept true detector and recognition NCHW
batches, and every implemented primitive uses the leading `N` dimension as
one dispatch dimension. This includes pointwise/residual, depthwise and
ordinary Conv, ConvTranspose, resize+Add, channel-affine, binary chains, and
recognizer GEMM. Immutable model weights/biases remain in the persistent
Vulkan context when an admitted segment repeats; dynamic images and residuals
are uploaded once per complete batch. The runtime restores compact descriptor
views before every dispatch, preventing a preceding cached parameter shape
from affecting a later batched operator.

Hybrid execution has one synchronized Vulkan queue, so parallel host graph
batches cannot run device work concurrently. `Options::hybrid_graph_parallelism`
therefore defaults to **1** (benchmark override
`PPOCR_HYBRID_GRAPH_PARALLELISM`), avoiding serialized submissions together
with multiplied CPU activation workspaces. CPU-only keeps
`rec_parallelism` independent batches in flight. This is a scheduling and
memory-peak safeguard, not a restriction on the tensor batch itself; use a
positive override only after measuring the target driver.

After the change, the AVX-512 kernel smoke and Radeon 780M Vulkan primitive
smoke both passed. On four same-shape tiny `en.ppm` pages with hybrid mode,
`det_batch_size=4`, `rec_batch_size=4`, four-way preprocessing, one warm-up,
and three timed runs, public OCR results were serial-identical. The queue-aware
default measured **150.969 / 105.663 ms** serial/batched (**1.429x**). An
explicit four-graph-worker A/B measured **131.838 / 102.749 ms** (**1.283x**)
on the same shared host. CPU-only under the same batch settings measured
**133.062 / 98.406 ms** (**1.352x**). These short, shared-host snapshots
demonstrate correct batch scheduling; they do not alter the strict
per-primitive GPU admission policy, do not claim universal GPU speedups, and
do not make `gpu_only` available.

### 2026-08-12: refreshed three-model and varied-size baseline

`ppocr_mem_soak` now accepts the same backend and bounded-batch environment
settings as the benchmark (`PPOCR_BACKEND`, detector/recognition batch sizes,
preprocessing, and hybrid graph parallelism). This makes the memory check
cover the production CPU/hybrid scheduler rather than always constructing a
default OCR instance.

With `PPOCR_APPROX_GELU=1`, CPU mode, two warm-ups and eight timed `en.ppm`
runs on this AVX-512 host, current C++ means were tiny **28.602 ms**, small
**58.717 ms**, and medium **206.668 ms**. Hybrid on the same short workload
measured **32.243 / 70.071 / 259.296 ms** respectively, so CPU was correctly
retained for this particular small-map run when Vulkan segment admission did
not improve the full OCR boundary. The earlier representative ORT comparison
still establishes a lead for tiny/small; medium remains an active optimization
target and is not claimed universally ahead of ONNX Runtime.

A medium hybrid varied-size one-cycle soak with seven samples (16x16,
English, low-contrast, ultra-wide, ultra-tall, dense 1920x1080, and
2560x1440 screenshot), `det_batch_size=4`, `rec_batch_size=4`, and four-way
preprocessing finished at **+2,134,016 bytes** over a
**436,551,680-byte** post-warmup private-byte baseline. This is a process
allocator high-water snapshot, not a leak proof; it demonstrates the current
test corpus and bounded activation behavior across materially different input
dimensions.

### 2026-08-12: queue-aware detector batching

`RecognizeBatch()` now applies the same queue-aware graph scheduling rule to
the detector as it already applied to recognition. `image_batch_parallelism`
still controls independent detector graphs in CPU mode. In hybrid mode, the
default is instead `hybrid_graph_parallelism=1`: one persistent synchronous
Vulkan queue cannot execute competing detector graphs concurrently, and
running several of them only retains several large detector activation
frontiers while their GPU submissions serialize. This does not reduce the
real tensor batch: same-resize pages still form bounded `[N,3,H,W]` detector
batches and all implemented Vulkan kernels map that N dimension to one
dispatch. Set `PPOCR_HYBRID_GRAPH_PARALLELISM` above one only after a
driver-specific A/B measurement.

Post-change AVX-512 kernel and Radeon 780M Vulkan primitive smoke both
passed. Four identical `en.ppm` pages, `det_batch_size=4`,
`rec_batch_size=4`, four-way preprocessing, two warm-ups and five timed runs
were serial-result-identical. With the hybrid default, tiny measured
**147.010 / 107.388 ms** serial/batched (**1.369x**) and medium measured
**1050.363 / 1026.150 ms** (**1.024x**). The matching CPU-only measurements
were **136.004 / 105.275 ms** (**1.292x**) for tiny and **994.335 / 976.100
ms** (**1.019x**) for medium. An explicit hybrid four-graph-worker experiment
was not a win in this sample: tiny batched latency rose to **158.929 ms**;
medium rose to **1096.940 ms**. A four-image varied-size medium hybrid smoke
also stayed serial-result-identical at **1242.143 / 1205.323 ms** (**1.031x**).
These shared-host measurements validate the scheduler and batch correctness;
they do not claim a universal GPU advantage. `gpu_only` remains unavailable
until the full PP-OCRv6 graph and post-processing are device-resident.

### 2026-08-12: medium CPU parallelism retune

The persistent CPU kernel executor's automatic ceiling is now **16** workers
(bounded by available hardware); `PPOCR_BENCH_THREADS` remains the explicit
deployment override. The previous automatic ceiling of 12 left measurable
medium-detector channel/tile parallelism unused on this AVX-512 host. Tiny and
small keep their short-kernel thresholds, so this is not a promise that every
model launches sixteen workers.

On the local AVX-512 host with `PPOCR_APPROX_GELU=1`, two warm-ups and twelve
timed `en.ppm` runs, the default CPU path measured tiny **28.124 ms**, small
**61.297 ms**, and medium **188.145 ms**. A medium 3000x80 strip measured
**130.692 ms** over eight runs. The controlled medium A/B was 12 workers:
**215.234 ms** mean, 16 workers: **189.517 ms** mean (both twelve-run
samples); 20 and 24 workers regressed, so the automatic cap is deliberately
16 rather than unbounded. The current same-host ORT eight-run `en.ppm`
snapshot is tiny **74.609 ms**, small **86.288 ms**, and medium **147.779
ms**: tiny/small retain clear native wins, while medium remains slower than
ORT on that comparison and is not claimed as meeting the final target.

The latest Radeon 780M Vulkan primitive suite still passed. Its full-boundary
admission selected GPU for the Swish channel-affine and depthwise-Swish batch
shapes, while ordinary Conv, pointwise Conv, GEMM and resize/add samples
remained transfer-bound and correctly retained CPU. A medium hybrid `en.ppm`
sample was **295.677 ms** (eight runs), versus the CPU result above, confirming
that a device-resident multi-operator graph segment閳ユ攺ot forced standalone
dispatches閳ユ攰s required before claiming a medium GPU end-to-end win.

### 2026-08-12: N=8 GPU-aware page batch and PPM stress gate

The public batch route was rechecked with eight identical tiny `en.ppm`
pages. It forms an exact `[8,3,H,W]` detector tensor and packs compatible
recognition crops into `[N,3,48,W]`; no page or crop is resampled merely to
fill a batch. Vulkan maps that leading N dimension to one compute-dispatch
dimension for each implemented hybrid primitive. With 16 CPU kernel workers,
`det_batch_size=8`, `rec_batch_size=8`, eight-way preprocessing, one graph
submitter, two warm-ups and eight timed runs, serial results and batch results
were exactly identical (16 decoded OCR results):

| Backend | Serial mean | N=8 batch mean | Batch speedup |
|---|---:|---:|---:|
| CPU | 337.102 ms | 233.977 ms | 1.441x |
| Hybrid Vulkan | 454.287 ms | 239.397 ms | 1.898x |

This is a same-host batch scheduling measurement, not a claim that the
partial Vulkan graph makes whole-image OCR faster than CPU on the Radeon 780M:
the hybrid N=8 mean was still 2.3% above the CPU N=8 mean. It does verify that
Vulkan-enabled batching is a real single-NCHW submission path with correct
output, rather than eight independent GPU calls.

### 2026-08-13: front-end address reduction and expanded recovery check

The fused RGB bilinear-resize/NCHW-normalization front end now hoists its two
source-row bases out of the destination-pixel loop.  This removes four
row/column address multiply-add chains per output pixel, without changing the
four source samples, interpolation order, integer rounding, or model input.
It applies to both detector-page resize and recognizer-crop resize paths, so
the benefit is architecture-neutral and composes with the existing AVX2,
AVX-512, and NEON graph kernels.

The change passed the complete low-level SIMD smoke suite and decoded-text
comparison with ONNX Runtime for all model sizes: tiny, small, and medium each
passed **4/4** inputs (`16x16`, English two-line, low-contrast text, and an
ultra-wide `3000x80` strip), for **12/12** total.  The wider robustness corpus
also passed **11/11** cases for tiny and small: blank, word, rotated text,
low-contrast text, wide/tall strips, dense blobs, dense UI, and 1920x1080 /
2560x1440 screenshots.  Medium passes the same semantic corpus except its
current detector threshold produces extra low-score components on the blank
and dense-blob synthetic cases; this is tracked as DB-postprocess tuning work,
not hidden as a performance success.

A fresh exact `N=4` CPU batch recheck (`det_batch_size=4`,
`rec_batch_size=4`, four-way preprocessing, one warm-up, five timed runs)
kept all eight decoded outputs serial-identical.  Mean batch speedups on this
shared AVX-512 Windows host were tiny **1.490x**, small **1.129x**, and medium
**1.243x**.  The medium value has visibly variable shared-host samples, so
these figures are evidence of the batched code path rather than portable
capacity guarantees.

The expanded `16x16 -> 64x64 -> 160x120 -> 320x240 -> 640x480 -> 1280x720
-> 1920x1080 -> 2560x1440 -> 64x64` memory ladder was rerun for medium.  CPU
passed two cycles from a 146,903,040-byte warmed private baseline and returned
to 143,212,544 bytes (-2.51%) and 144,183,296 bytes (-1.85%); private bytes
and working set both declined after the largest image in both cycles.  Hybrid
Vulkan also passed: its 392,998,912-byte baseline returned to 393,195,520
bytes (+0.050%) in both cycles.  Hybrid private bytes remained 0.0031% above
the immediately preceding largest sample while working set declined, which is
expected persistent Vulkan/Windows allocator capacity rather than retained
image or activation memory.

### 2026-08-13: selective SIMD GELU dispatch

The opt-in `PPOCR_APPROX_GELU=1` path now reaches independent generic
`FusedGelu` activations as well as the existing fused BatchNorm+GELU route.
On x86 it dynamically selects AVX-512 first, then AVX2; the baseline exact
Erf implementation and ARM/NEON fallback remain unchanged.  The approximate
Pad鑼?tanh polynomial is intentionally limited to broad activations of at
least 262,144 elements.  That preserves the established exact path for the
short, decision-sensitive tiny/small recognizer tensors while accelerating
the large medium feature maps. `PPOCR_DISABLE_APPROX_GELU_SIMD=1` is a direct
deployment A/B escape hatch.

The AVX2 generic GELU kernel was also made genuinely vectorized: it no longer
stores each vector and calls scalar `tanh` eight times. AVX2 and AVX-512 now
use the same bounded Pad鑼?approximation for vector bodies and scalar tails.
The low-level smoke verifies separate and in-place large/tail GELU buffers,
and passed in both approximate and default exact modes.

With 16 workers on the local AVX-512 host, a paired medium CPU `en.ppm` A/B
(two warm-ups, eight timed runs) improved from **216.867 ms** with the new
generic SIMD route disabled to **205.715 ms** enabled (**5.1% lower** mean
latency).  Tiny, small and medium each again passed all four decoded-text
comparisons against ONNX Runtime (`16x16`, English, low contrast and
`3000x80`), **12/12** total.  The same-host 12-run CPU snapshot after the
change was tiny **34.335 ms** and small **75.072 ms**; shared-host variance
means these are compatibility snapshots, not portable throughput guarantees.

### 2026-08-13: SIMD terminal-CTC memory route

`PPOCR_ENABLE_FUSED_TERMINAL_CTC=1` now computes each reusable terminal-logit
row through the normal runtime-dispatched GEMM kernel instead of a separate
scalar `K鑴砎` loop. It still avoids materializing the full terminal `[N,T,V]`
logits tensor, but now uses AVX-512, AVX2, or ARM/NEON GEMM as available and
uses the existing ISA argmax. The K-order, selected-class Softmax calculation,
blank/repeat CTC recurrence, and output confidence remain unchanged.

With this opt-in mode and 16 CPU workers, all three models passed the four
diverse ONNX Runtime decoded-text inputs (**12/12**). The new N=4 public batch
smoke was serial-result-identical and measured **1.367x** (tiny), **1.149x**
(small), and **1.188x** (medium) batch speedup on this shared AVX-512 Windows
host. Two warm-ups/eight English-page runs measured **35.933 / 74.952 /
246.148 ms** for tiny/small/medium. The medium two-cycle
`16x16 閳?閳?閳?2560x1440 閳?64x64` CPU memory ladder also passed: a
**147,066,880-byte** warmed baseline returned to **143,347,712 bytes**
(-2.53%) and **147,116,032 bytes** (+0.03%), both below their pre-return
cycle high-water marks. The regular full-terminal-logits SIMD route remains
the default; this stays an explicit memory/performance deployment option.

### 2026-08-13: four-row terminal-CTC GEMM tile

The optional terminal-CTC route now projects up to four adjacent time steps
into its bounded vocabulary workspace at once, then performs the CTC scan in
the original time order. This keeps memory bounded to `4 鑴?vocab` values per
active sequence rather than `[N,T,V]`, while allowing the AVX-512/AVX2 GEMM
microkernels to reuse immutable projection weights across rows. The final
partial tile and every blank/repeat transition retain the same semantics.

On the 16-worker AVX-512 host, two warm-ups/eight `en.ppm` runs with
`PPOCR_ENABLE_FUSED_TERMINAL_CTC=1` measured **32.647 / 70.079 / 226.250 ms**
for tiny/small/medium. Compared with the preceding one-row fused-CTC sample,
this is **9.1% / 6.5% / 8.1%** lower latency. The post-change N=4 CPU smoke
remained serial-result-identical for all models and measured **1.350x**,
**1.201x**, and **1.195x** batch speedup respectively. The low-level kernel
smoke, all three four-image ONNX Runtime decoded-text comparisons (**12/12**),
and Radeon 780M Vulkan primitive/admission smoke passed. As intended, hybrid
continues to select GPU only for shapes that win the full transfer boundary;
the terminal CTC tile is a CPU-memory optimization, not a GPU-only path.

### 2026-08-13: four-row terminal-CTC resolution-ladder memory recheck

The deterministic `16x16 -> 64x64 -> 160x120 -> 320x240 -> 640x480 ->
1280x720 -> 1920x1080 -> 2560x1440 -> 64x64` ladder was rerun after the
four-row terminal-CTC tile, with `PPOCR_BENCH_THREADS=16`,
`PPOCR_APPROX_GELU=1`, and `PPOCR_ENABLE_FUSED_TERMINAL_CTC=1`. The CPU
medium model passed the two-cycle bounded-recovery gate. Its warmed private
baseline was **143,339,520 bytes**; the post-large `64x64` samples were
**148,660,224 bytes** (**+3.71%**) and **147,742,720 bytes** (**+3.07%**).
They were only **+0.68%** above the immediately preceding `2560x1440`
sample in each cycle, well below the 20% gate.

In this particular shared-Windows-host run, neither private bytes nor working
set immediately declined after the largest page: the allocator retained or
reused capacity, rather than keeping the RGB image or activation objects
alive. The checker records this explicitly as `declined_after_largest=false`
and retains every per-resolution private-byte/working-set sample in
`build/memory_ladder_medium_fused_ctc_tile4_cpu_20260813.json`; it therefore
does not mislabel stable allocator capacity as immediate memory release. The
completed image is destroyed before every sample, and the bounded-recovery
pass is the portable regression condition.

### 2026-08-13: AVX-512 reconstruction 2x2 four-output default

The CPU dispatcher now enables the existing AVX-512 four-output tile for
valid `2x2` reconstruction convolutions by default when the shape has at
least eight input channels. It reuses each 16-float input vector across four
output filters while retaining the same per-output reduction order, scalar
tail, AVX2, NEON, and baseline fallbacks. This covers the tiny/small/medium
detector reconstruction pairs (`8閳?6`, `12閳?4`, and `32閳?4` channels) without
introducing an ISA requirement. Set `PPOCR_DISABLE_AVX512_CONV2X2_TILE4=1`
for a deployment A/B fallback.

On the local shared AVX-512 Windows host (`PPOCR_BENCH_THREADS=16`, opt-in
approximate GELU and four-row terminal CTC), the medium English page improved
from **237.935 ms** to **230.035 ms** across 16 timed runs (**3.3% lower**
mean latency). A two-run dense `1920x1080` sample also reduced from
**4,740.695 ms** to **4,208.679 ms** (**11.2% lower**); this dense figure is
intentionally short and host-variable, so it is directional rather than a
portable throughput guarantee. A fresh 2-warmup/8-run CPU snapshot measured
tiny **35.970 ms**, small **79.078 ms**, and medium **235.396 ms** on `en.ppm`.

All three models passed the four varied ONNX Runtime decoded-text cases
(**12/12**: `16x16`, English, low contrast, and `3000x80`) with the new
default. The N=4 public batch smoke stayed serial-result-identical and
measured **1.314x**, **1.210x**, and **1.087x** speedup for tiny, small, and
medium. The low-level exact/approximate SIMD smoke and the Radeon 780M Vulkan
admission smoke also passed. A current short hybrid sample was **35.627 /
84.529 / 294.957 ms** (tiny/small/medium), so the transfer-inclusive admission
rule continues to select Vulkan only for segments that actually meet the
no-slower condition; this does not claim complete GPU-only PP-OCRv6 support.

### 2026-08-13: terminal-projection SIMD coverage expansion

The dependency-free kernel smoke now includes the exact terminal projection
geometries of every shipped recognizer: `4x80x6906` for tiny and
`4x120x18710` / `4x192x18710` for small/medium, plus the bounded terminal-CTC
route. This explicitly exercises wide vocabulary tails, bias initialization,
the AVX-512 four-row GEMM reuse tile, and the unchanged scalar/AVX2/NEON
fallback contract rather than inferring coverage from a smaller synthetic
matrix.

A renewed CPU A/B on the English page (three warm-ups, 12 runs, 16 workers,
exact terminal logits) measured tiny/small/medium at **28.508 / 62.702 /
192.242 ms** with the default reconstruction tile, versus **28.950 / 63.142 /
198.772 ms** with `PPOCR_DISABLE_AVX512_CONV2X2_TILE4=1` (**1.5% / 0.7% /
3.3% lower**). Separately disabling the already-default AVX-512 four-row
GEMM tile produced **30.019 / 63.584 / 196.993 ms**, confirming that its main
measured benefit is medium's wide transformer projections. These remain
shared-host snapshots, not cross-machine guarantees.

### 2026-08-13: batched AVX-512 2x2 reconstruction tile

`Conv2dBatch` now uses the same AVX-512 four-output `2x2` valid-convolution
tile as the single-image dispatcher. Its scheduler owns one
`[batch, output-channel-tile]` task instead of one output plane, preserving
the exact per-channel reduction order while reusing every loaded input vector
over four filters for detector batches. The existing AVX2, NEON, scalar, tail,
and `PPOCR_DISABLE_AVX512_CONV2X2_TILE4=1` fallback paths remain intact.

With four equal English pages, `det_batch_size=4`, `rec_batch_size=4`,
four-way preprocessing, two warm-ups, and five timed runs, serial-vs-batch
results remained bit-identical. The tile-enabled batch means were **91.918 /
213.696 / 781.774 ms** for tiny/small/medium, versus **93.346 / 223.177 /
939.356 ms** with the tile disabled: **1.5% / 4.2% / 16.8% lower** batch
latency on this shared AVX-512 host. The corresponding complete
serial-to-batch speedups were **1.290x / 1.194x / 1.096x**. All three models
also passed their four-image ONNX Runtime decoded-text checks (**12/12**).

### 2026-08-13: refreshed small-to-large memory ladder

The native memory regression now exercises one live medium PP-OCRv6 instance
over a deterministic ascending image sequence: `16x16 -> 64x64 -> 160x120 ->
320x240 -> 640x480 -> 1280x720 -> 1920x1080 -> 2560x1440 -> 64x64`.  Each
image is decoded, inferred, destroyed, and only then sampled.  The report
therefore records process private bytes and Windows working-set bytes for
every resolution, plus whether the final small image returns below the
pre-large high-water mark and whether it immediately declines from the
preceding largest-image sample.

A fresh CPU run with 16 workers and `PPOCR_APPROX_GELU=1` passed two cycles
from a **143,192,064-byte** warmed private baseline.  The final `64x64`
samples were **148,197,376 bytes** (+3.50%) and **147,365,888 bytes**
(+2.91%); both were below the respective pre-return peaks of **149,053,440**
and **150,142,976 bytes**, and comfortably passed the 20% bounded-recovery
gate.  In this particular shared-Windows-host measurement, neither final
sample immediately fell below its directly preceding `2560x1440` sample
(+1.67% / +1.76%), and working set showed the same allocator-capacity
retention.  This is reported as `declined_after_largest=false`, not presented
as an immediate decommit: the completed large image and its inference
temporaries have already been destroyed before measurement.  The durable
per-resolution report is
`build/memory_ladder_medium_cpu_tile4_20260813.json`.

### 2026-08-13: tiny resolution-ladder recovery rerun

The same native small-to-large-to-small memory gate was also run against the
tiny PP-OCRv6 model with 16 CPU workers and `PPOCR_APPROX_GELU=1`.  It uses
the deterministic sequence `16x16 -> 64x64 -> 160x120 -> 320x240 ->
640x480 -> 1280x720 -> 1920x1080 -> 2560x1440 -> 64x64`, sampling only after
each decoded image and its inference temporaries have gone out of scope.

Both cycles passed the 20% bounded-recovery gate from a **9,195,520-byte**
warmed private-byte baseline.  Immediately after `2560x1440`, the final
`64x64` sample fell from **9,940,992** to **9,232,384 bytes** (**-7.13%**) in
cycle one, and from **10,711,040** to **9,330,688 bytes** (**-12.89%**) in
cycle two.  Working set declined in both cycles as well.  This directly
confirms recovery after the largest image on the current host; the durable
per-resolution report is `build/memory_ladder_tiny_cpu_latest.json`.

### 2026-08-13: Vulkan terminal-GELU chain and immutable-chain cache

The Vulkan MobileNet `depthwise -> pointwise` chain now has an optional mode
that applies the existing opt-in AVX2/AVX-512 GELU approximation in its final
GPU writeback.  It avoids the former GPU readback followed by a full CPU GELU
pass for the medium-only graph fusion.  This path is selected only when
`PPOCR_APPROX_GELU=1` and the CPU dispatcher would make that same approximate
choice for the same logical tensor size; exact ONNX Erf-GELU, ARM/NEON, and
small tensors keep their established CPU behavior.  The chain also reuses its
immutable depthwise weights, bias, and packed pointwise parameters across
consecutive submissions, while invalidating that cache whenever a shared
Vulkan scratch binding is overwritten.

The AMD Radeon 780M Vulkan smoke validates both regular and GELU-fused chain
outputs against the shared CPU kernels.  Hybrid admission still measures the
complete upload, dispatch, fence, and readback boundary and only chooses GPU
when it is no slower; on this host the representative depthwise-pointwise
chain remained CPU-preferred (**0.601 ms GPU / 0.183 ms CPU**), while
affine+Swish and depthwise+Swish primitives were admitted in the same smoke.
This is intentional: the change makes a qualifying GPU segment cheaper
without relaxing the no-slower policy.

With 16 workers and `PPOCR_APPROX_GELU=1`, decoded-text parity versus ONNX
Runtime passed all four diverse inputs for each official model (**12/12**):
`16x16`, English, low contrast, and `3000x80` ultra-wide text.  The paired
eight-run `en.ppm` CPU/hybrid snapshots were tiny **32.042 / 34.855 ms**,
small **63.781 / 70.924 ms**, and medium **191.886 / 219.099 ms**.  Thus this
UMA run correctly retains CPU for the complete models; it is not presented as
a blanket GPU speed claim.  Reports are
`build/accuracy_{tiny,small,medium}_vulkan_gelu_fusion_20260813.json` and
`build/bench_{tiny,small,medium}_cpu_hybrid_gelu_fusion_20260813.json`.

The two-cycle medium `16x16 -> ... -> 2560x1440 -> 64x64` CPU memory ladder
also passed from a **147,148,800-byte** warm baseline.  Private bytes fell
immediately after the largest image in both cycles (**-2.01%** and **-1.39%**)
and working set fell as well; the durable report is
`build/memory_ladder_medium_cpu_vulkan_gelu_fusion_20260813.json`.

### 2026-08-13: refreshed ascending-resolution memory measurement

A fresh two-cycle medium CPU run (16 kernel workers,
`PPOCR_APPROX_GELU=1`) reconfirmed the deterministic
`16x16 -> 64x64 -> 160x120 -> 320x240 -> 640x480 -> 1280x720 -> 1920x1080
-> 2560x1440 -> 64x64` sequence.  The checker samples process private bytes
and Windows working set only after each source image and its inference result
have left scope.  It passed the 20% bounded-recovery gate from a
**143,245,312-byte** warmed private baseline.

In cycle 1 the final `64x64` sample was **146,501,632 bytes** (+2.27% from
baseline, -0.63% from the pre-return private-byte peak); in cycle 2 it was
**143,319,040 bytes** (+0.05%, -4.66%).  The returned small image was below
the pre-return private-byte and working-set peaks in both cycles.  It was
**1.89%** above the immediately preceding `2560x1440` private-byte sample in
cycle 1, but **2.04%** below it in cycle 2; the working set follows the same
pattern.  Therefore the report truthfully records
`declined_after_largest=false` for cycle 1 and `true` for cycle 2 instead of
claiming unconditional immediate decommit.  This distinguishes bounded
live-memory recovery from Windows allocator capacity retention.  Full
per-resolution evidence is in
`build/memory_ladder_medium_cpu_latest_20260813.json`.

### 2026-08-13: SE-gate fusion investigation

The detector's largest squeeze-excitation block (`ReduceMean -> 1x1 -> ReLU
-> 1x1 -> HardSigmoid -> channel Mul`) was profiled as a large-page memory and
traffic candidate.  A strict graph fusion was added behind
`PPOCR_ENABLE_SE_GATE_FUSION=1`: it retains the dying NCHW source allocation,
forms only compact `[N,C,1,1]` gate storage, then scales the source in place.
It preserves the original per-channel FP32 reduction and gate order, and the
tiny/small/medium text regression passed **12/12** cases with it enabled.

It is intentionally **not** a production default yet.  On the current
16-worker AVX-512 host, the 30-run `en.ppm` A/B was favourable for tiny
(**30.926 -> 29.772 ms**, 1.039x), but regressed small (**61.034 -> 63.710
ms**) and was neutral for medium (**205.271 -> 205.028 ms**).  A three-run
medium `2560x1440` page A/B was also within noise but not a reliable win
(**2924.075 -> 2952.852 ms**).  Retaining this as opt-in prevents a
memory-oriented fusion from silently violating the all-three-model latency
requirement; the exact gate pattern remains available for a future
device-resident Vulkan chain.

After this investigation, a fresh medium two-cycle resolution ladder again
passed the 20% bounded-recovery gate.  Its warmed private-byte baseline was
**143,261,696 bytes**.  The final `64x64` sample fell below the immediately
preceding `2560x1440` sample in cycle 1 (**-1.70%**) and remained within the
gate in cycle 2 (**+2.13%**); working-set results and every resolution are in
`build/memory_ladder_medium_cpu_post_se_experiment_20260813.json`.

### 2026-08-13: AVX-512 stride-two NCHW batch tile

The AVX-512 `3x3, stride=2` NCHW batch scheduler now assigns one task to a
four-output-channel tile instead of invoking the kernel one channel at a
time.  Its SIMD body therefore shares every gathered/packed source vector
across four filters, exactly as the single-image dispatcher already did.
Output-channel ownership remains disjoint and each output retains the same
ascending FP32 accumulation order.  The optimization is dynamically entered
only after the existing AVX-512 runtime feature check; AVX2, scalar, and ARM
NEON paths are unchanged.  Set
`PPOCR_DISABLE_AVX512_STRIDE2_BATCH_TILE4=1` for a deployment A/B fallback.

The low-level kernel smoke and Vulkan smoke passed.  Decoded-text comparison
against ONNX Runtime passed all four varied inputs (`16x16`, English,
low-contrast, `3000x80`) for tiny, small, and medium (**12/12**); reports are
`build/accuracy_{tiny,small,medium}_stride2_batch_tile4_20260813.json`.

On four equal English pages, real detector and recognizer NCHW batches
(`det_batch=4`, `rec_batch=4`, exact width, two warm-ups, twelve timed runs)
remained serial-result-identical and measured these complete
serial-to-batch speedups on the 16-worker AVX-512 host: **1.573x** tiny
(184.036 / 116.981 ms), **1.379x** small (407.445 / 295.523 ms), and
**1.279x** medium (1105.502 / 864.359 ms).  This exercises the new stride-two
batch route directly, rather than inferring a batch benefit from a single
page.  Exact sample records are
`build/batch_{tiny,small,medium}_stride2_batch_tile4_20260813.txt`.

The same refreshed eight-run single-English CPU/Hybrid check measured
CPU **33.938 / 71.164 / 206.282 ms** and Hybrid
**36.744 / 74.706 / 292.331 ms** for tiny/small/medium.  On this UMA Radeon
780M host, Hybrid remains slower end-to-end, while individual admitted
Vulkan affine+Swish and depthwise+Swish segments still obey the
transfer-inclusive no-slower policy.  The result is not presented as a
whole-model GPU speed claim.

### 2026-08-13: heterogeneous-size batch regression

`bench/heterogeneous_batch_regression.py` adds a repeatable multi-size page
batch gate around `ppocr_batch_smoke`.  Its PPM corpus deliberately combines
`16x16`, ordinary two-line text, low-contrast text, `3000x80` wide text,
`80x3000` portrait text, a dense `1920x1080` UI page, and a `2560x1440`
screenshot.  It executes both exact-width (`PPOCR_REC_WIDTH_BUCKET=1`) and
a candidate padded-width grouping, while `ppocr_batch_smoke` verifies every
batch result is identical to serial inference.  The JSON records both timing
samples rather than promoting a padding policy from a narrow equal-page test.

On the current 16-worker AVX-512 CPU with approximate GELU, all three models
passed serial-identity on this seven-image / 186--188-result corpus.  The
exact-width batch means were **473.748 ms** (tiny), **1771.887 ms** (small),
and **8437.291 ms** (medium), with serial-to-batch speedups of **1.258x**,
**1.156x**, and **1.038x**, respectively.  A 16-pixel candidate bucket
measured **487.901 ms** (+3.0%), **1796.819 ms** (+1.4%), and **8945.280 ms**
(+6.0%).  Exact-width therefore remains the lower-memory CPU default; the
candidate remains an exposed deployment knob for workloads where reduced
shape fragmentation pays back its bounded right-padding.  Reports are in
`build/heterogeneous_batch_{tiny,small,medium}_bucket16_20260813.json`.

### 2026-08-13: batch-invariant GELU policy

The selective GELU dispatch now receives the logical per-sample element count
from every generic and fused graph route.  Consequently, packing four equal
pages/crops into an NCHW batch cannot change a short tensor from exact Erf to
the opt-in approximate formula merely because the batch has four times as many
elements.  This closes a batch-correctness edge case while keeping the medium
large-map AVX-512/AVX2 acceleration.  The low-level GELU smoke passed in both
exact and approximate configurations, and all three models again passed the
four-image ONNX Runtime decoded-text gate (**12/12**).

An N=4 CPU public-batch smoke with `det_batch_size=4`, `rec_batch_size=4` and
four-way preprocessing remained serial-result-identical for every model. On
this shared AVX-512 host it measured tiny **1.346x**, small **1.090x**, and
medium **1.055x** serial-to-batch speedup.  A new paired medium CPU A/B (two
warm-ups, eight runs) measured **211.217 ms** with the corrected SIMD route
and **219.778 ms** with `PPOCR_DISABLE_APPROX_GELU_SIMD=1` (**3.9%** lower
mean latency). Hybrid measured **314.123 ms** in the same short sample; the
strict transfer-inclusive Vulkan admission therefore remains necessary and
continues to retain CPU for transfer-bound operators.

### 2026-08-13: Vulkan batch-admission parity correction

The per-shape Vulkan admission probes for pointwise Conv and residual
pointwise Conv now benchmark the same batched CPU SIMD kernels used by live
`[N,C,H,W]` OCR execution.  Previously these two probes used a serial loop of
single-image kernels even though the executor flattens batch/channel tiles;
that could make a GPU segment appear no slower than a weaker CPU reference.
The correction does not change model arithmetic or force a GPU path. It makes
the hybrid policy stricter and accurate: Vulkan is selected only when its full
upload, dispatch, synchronization, readback, and validation boundary is no
slower than the real batched CPU alternative.

`ppocr_vulkan_smoke` and the full SIMD kernel smoke passed on the local AMD
Radeon 780M / AVX-512 host.  The current primitive measurements correctly
retain CPU for pointwise Conv (`0.22340 ms` GPU versus `0.02218 ms` CPU) and
residual pointwise Conv+ReLU (`0.24984 ms` versus `0.03832 ms`), while keeping
GPU for the validated depthwise+Swish sample (`0.41966 ms` versus `0.42462
ms`).  This is explicit evidence of the requested no-slower hybrid policy,
not a claim of universal GPU speedup or GPU-only availability.

### 2026-08-13: current three-model CPU/hybrid snapshot

With `PPOCR_BENCH_THREADS=16` and `PPOCR_APPROX_GELU=1`, a refreshed CPU
measurement on the two-line English PPM used two warm-ups and twelve timed
runs.  Tiny measured **33.025 ms** (30.280 FPS), small **70.530 ms** (14.178
FPS), and medium **246.420 ms** (4.058 FPS).  A controlled medium worker scan
on the same shared host measured 8 / 12 / 16 workers at **366.673 / 282.361 /
246.420 ms**, respectively; this validates the current 16-worker ceiling for
the broad medium graph without claiming that all deployments share that curve.

Hybrid mode with the same model/input and two warm-ups/eight runs measured
tiny **37.400 ms**, small **75.953 ms**, and medium **235.987 ms**.  Medium
therefore saw a **4.2%** lower mean latency on this local hybrid run, while
tiny/small correctly remain CPU-preferred at the segment level when transfers
do not pay back.  A paired medium ONNX Runtime 1.28 CPUExecutionProvider run
(matching 16-thread budget, two warm-ups/eight runs) measured **399.188 ms**;
the native medium CPU snapshot is **38.3%** lower latency and hybrid is
**40.9%** lower on this host.  These are shared-host measurements rather than
portable guarantees; model accuracy and the hybrid admission rule remain
independent of benchmark mode.

`bench/stress_regression.py` now resolves a default `.png` case to a sibling
`.ppm` case when necessary. The checked-in native regression corpus can
therefore run directly without duplicating the large source images or adding
an image decoder to the C++ executable. The tiny-model CPU stress gate passed
all **11/11** cases: dense UI, blank, single word, ultra-wide/tall strips,
tiny, low-contrast, rotated, dense blobs, and two screenshot sizes.

### 2026-08-13: Vulkan GELU-chain smoke repair and refreshed checks

The Vulkan smoke's depthwise-to-pointwise mode 30 test now calculates the
same explicit Pad鑼?tanh GELU approximation as the shader. Previously the test
accidentally compared this opt-in approximation with exact `kernels::Gelu()`
when `PPOCR_APPROX_GELU` was unset, producing a false failure even though the
shader's mode was correct. This is a test-only correction: it does not change
the default exact-GELU execution policy or GPU admission rule. The rebuilt
AVX-512 kernel smoke and Radeon 780M Vulkan primitive suite passed, including
the device-resident depthwise閳姫ointwise閳墱ELU chain.

The refreshed CPU-versus-ORT `en.ppm` comparison (16 kernel workers, opt-in
approximate GELU, 2 warm-ups/20 timed runs) measured native tiny **40.480 ms**
versus ORT **122.008 ms** (**66.8% lower latency**), small **79.199 ms** versus
**183.173 ms** (**56.8% lower**), and medium **242.245 ms** versus **291.551
ms** (**16.9% lower**).  These are local shared-host measurements, not a
portable guarantee. Decoded-text parity against ORT also passed four diverse
inputs for each model (English, mixed text, single-line text, and low-contrast
multi-line text): **12/12**.

Hybrid retains the strict full-boundary admission rule and therefore only
uses a Vulkan segment when its measured H2D閳妿ispatch閳姁ence閳墬2H result is no
slower than the matching CPU SIMD work. A seven-image varied-size hybrid batch
(16x16, English, low contrast, ultra-wide, ultra-tall, 1920x1080 UI, and
2560x1440 screenshot) stayed serial-result-identical and took **850.547 ms**
batched versus **1049.532 ms** serial (**1.234x**) with four graph workers.
A two-cycle tiny Hybrid resolution ladder also passed every bounded memory
recovery limit; its final `64x64` samples stayed below the warmed
119,021,568-byte private baseline. `gpu_only` remains deliberately rejected:
the complete OCR graph and DB post-processing are not yet device-resident, so
silently using a partial GPU path would violate the public backend contract.

### 2026-08-13: refreshed ascending-resolution memory recovery check

The repository includes a reproducible ascending-resolution PPM ladder and a
native process-memory checker: `bench/generate_memory_ladder.py` creates
`16x16 -> 64x64 -> 160x120 -> 320x240 -> 640x480 -> 1280x720 -> 1920x1080
-> 2560x1440 -> 64x64`; `ppocr_mem_soak` loads, infers, and destroys each
image before sampling Windows private bytes and working set.  The final
`64x64` sample is therefore a direct observation after the largest image,
rather than a retained source-image allocation.  The checker requires the
returned small-image private bytes to stay within 20% of the warmed baseline,
the preceding `1920x1080` sample, and the immediately preceding largest-page
sample; it additionally records whether private bytes and working set actually
declined after `2560x1440`.

A fresh two-cycle CPU run with `PPOCR_BENCH_THREADS=16` and
`PPOCR_APPROX_GELU=1` passed for both available dictionary/model pairs. Tiny
returned from **10,756,096 B** at `2560x1440` to **10,199,040 B** (-5.18%) in
cycle 1 and from **9,953,280 B** to **9,420,800 B** (-5.35%) in cycle 2;
private memory therefore declined directly after the large page in both
cycles. Medium returned from **146,243,584 B** to **143,228,928 B** (-2.06%)
in cycle 1; cycle 2's returned sample was **146,063,360 B**, 1.58% above that
cycle's immediate `2560x1440` sample but still below its pre-return peak and
0.62% below the warmed baseline. Both models met every bounded-recovery gate.
The complete per-resolution private-byte and working-set records are retained
in `build/memory_ladder_tiny_cpu_resolution_recheck_20260813.json` and
`build/memory_ladder_medium_cpu_resolution_recheck_20260813.json`. These are
host-specific allocator observations, not a guarantee that every platform will
immediately decommit heap capacity after a large image.

### 2026-08-13: SE workspace reuse and refreshed medium memory ladder

`SqueezeExcitationGateInplace` now stores one `[N,C]` gate/mean buffer and a
single reusable `[reduced-C]` hidden row.  The former implementation retained
two `[N,C]` temporary buffers plus `[N,reduced-C]`; the helper's temporary
workspace is consequently reduced from `2*N*C + N*reduced-C` to
`N*C + reduced-C` floats, without changing its FP32 accumulation order or
the completed gate values.  This is a local SE-workspace reduction, not a
claim that total process memory drops by the same amount.

The rebuilt SIMD and Vulkan smoke suites passed, and decoded-text parity with
ONNX Runtime remained **4/4 for each of tiny, small, and medium** across
English, mixed, single-line, and low-contrast inputs.  The refreshed tiny and
small reports are
`build/accuracy_{tiny,small}_cpu_se_workspace_20260813.json`; medium was split
into `build/accuracy_medium_cpu_se_workspace_20260813_part{1,2}.json` to keep
each check bounded.  A fresh two-cycle CPU ascending-resolution run
(`16x16 -> ... -> 2560x1440 -> 64x64`, 16 workers, approximate GELU) also
passed every 20% bounded-recovery gate.  Its warmed private baseline was
**147,709,952 B**; the final small image was **143,351,808 B** and
**143,364,096 B**, respectively.  Private bytes and working set both declined
directly after `2560x1440` in both cycles.  Per-resolution private and working
set samples are retained in
`build/memory_ladder_medium_cpu_se_workspace_20260813.json`.

### 2026-08-13: extended-resolution and aspect-ratio memory check

The reproducible ladder now also covers `3000x1600`, `3840x2160`, the
ultra-wide `4096x256`, and ultra-tall `256x4096` before returning to `64x64`.
This exercises both pixel-count growth and detector resize branches that a
landscape-only corpus misses.  On the local 16-worker CPU, both tiny and
medium passed two cycles of this extended sequence.  For medium the
143,294,464-byte warmed baseline returned to **143,347,712 B** (+0.04%) and
**143,392,768 B** (+0.07%); both samples were below the immediately preceding
`3840x2160` page and their cycle high-water marks, and working set declined as
well.  Tiny returned to **10,211,328 B** (+1.47%) and **9,596,928 B** (-4.64%)
from its 10,063,872-byte baseline; its private bytes declined after the 4K
page in both cycles.  Its first-cycle working set did not immediately decline
from the preceding 4K sample, but remained below the cycle high-water mark;
the second cycle did decline.  Full durable resolution-by-resolution reports are
`build/memory_ladder_{tiny,medium}_cpu_extended_aspect_20260813.json`.

### 2026-08-13: staged post-4K memory recovery check

The memory ladder now continues after `3840x2160` with `1920x1080`,
`4096x256`, and `256x4096` before its final `64x64` sample. This distinguishes
recovery after a large area from recovery after the detector's two extreme
aspect-ratio resize paths. A fresh medium CPU pass (16 workers, approximate
GELU) warmed at **144,003,072 B** private bytes, peaked at
**149,565,440 B**, and returned to **144,084,992 B** after the full tail
(+0.06% over the warm baseline; -3.31% versus the 4K sample). Its working set
also returned from the 4K **152,662,016 B** to **147,812,352 B**. The durable
per-resolution report is
`build/memory_ladder_medium_cpu_staged_recovery_20260813.json`. These are
local allocator observations, not a cross-platform decommit guarantee.

### 2026-08-13: selective default SE graph fusion

SE gate fusion is now enabled by default only for detector blocks with at
least 512 input channels, which selects the three high-cost medium blocks and
keeps tiny/small on their lower-overhead established paths.  The full-shape
experiment remains available through `PPOCR_ENABLE_SE_GATE_FUSION=1`; disable
all default fusion with `PPOCR_DISABLE_SE_GATE_FUSION=1`, or tune the boundary
with `PPOCR_SE_GATE_FUSION_MIN_CHANNELS`.  This is a genuine graph/operator
fusion: it replaces `ReduceMean -> Conv -> Relu -> Conv -> HardSigmoid -> Mul`
with one in-place gate operation, avoiding the broadcast-map destination and
the intermediate graph dispatches.

On the same shared 16-worker AVX-512 host, a 2-warm-up/20-run English-page
CPU snapshot measured **30.982 ms** (tiny), **64.365 ms** (small), and
**195.898 ms** (medium), respectively.  The medium run before this selective
default measured 210.198 ms in a separate 12-run sample; this comparison is
directional because the host is shared, rather than a portable performance
claim.  The existing 20-run CPU-versus-ORT measurements remain the published
cross-engine evidence.  Kernel and Vulkan smoke both passed, and medium
retained ONNX Runtime decoded-text parity on all four diverse images; reports
are `build/bench_{tiny,small,medium}_cpu_selective_sefusion_20260813.json`
and `build/accuracy_medium_cpu_selective_sefusion_20260813_part{1,2}.json`.

### 2026-08-13: folded Conv channel-bias execution path

The graph loader already recognizes the recognizer pattern `Conv ->
Add([1,C,1,1]) -> Identity` and precomputes the resulting per-output-channel
bias. Its executor now passes that folded bias directly to the convolution,
removing the accidental second `AddChannelBias` traversal that had remained
after load-time folding. This keeps the original FP32 order閳ユ攦onvolutional
reduction followed by the same accumulator bias閳ユ攣nd avoids a full NCHW
read/write for each matched node. Set `PPOCR_DISABLE_FOLDED_CHANNEL_BIAS=1`
for a runtime A/B comparison.

All three model sizes retained decoded-text parity with ONNX Runtime on the
four-image English/mixed/single-line/low-contrast corpus (12/12 cases):
`build/accuracy_{tiny,small,medium}_cpu_folded_channel_bias_20260813.json`.
Kernel and Vulkan smoke both passed. On this shared 16-worker AVX-512 host,
a 2-warm-up/16-run CPU snapshot measured **31.574 ms** (tiny), **67.060 ms**
(small), and **201.835 ms** (medium); the matching files are
`build/bench_{tiny,small,medium}_cpu_folded_channel_bias_20260813.json`.
Separate 40-run A/B samples showed a medium reduction from **211.568 ms** to
**201.835 ms** (4.6%), while tiny/small were within normal shared-host noise;
this is therefore retained as a medium-oriented optimization rather than a
claim of uniform speedup on every model or system.

The staged medium CPU memory test also passed after the change:
**143,548,416 B** warmed private bytes, **151,822,336 B** at 4K, and
**143,601,664 B** at the final `64x64` image (+0.04% versus warm; -5.41%
versus 4K). Working set fell from **153,632,768 B** at 4K to
**147,640,320 B**. The durable report is
`build/memory_ladder_medium_cpu_folded_channel_bias_20260813.json`.

### 2026-08-13: refreshed full-resolution memory recovery measurement

The reproducible native memory ladder was rerun after rebuilding the current
CPU/Vulkan sources.  It processes one decoded RGB image at a time in this
order: `16x16 -> 64x64 -> 160x120 -> 320x240 -> 640x480 -> 1280x720 ->
4096x256 -> 256x4096 -> 1920x1080 -> 2560x1440 -> 3000x1600 -> 3840x2160 ->
1920x1080 -> 4096x256 -> 256x4096 -> 64x64`.  Thus the last small image is
sampled only after the largest-area page and both detector extreme-aspect-ratio
paths have completed and their input images have been destroyed.

With medium PP-OCRv6, CPU backend, 16 kernel workers, and approximate GELU,
the warmed private baseline was **144,879,616 B**.  The `3840x2160` sample was
**151,138,304 B** and the pre-return high-water mark was **152,875,008 B**;
the final `64x64` sample fell to **143,867,904 B** (-0.70% from warm, -4.81%
from the 4K sample, and -5.89% from the cycle high-water mark).  Its Windows
working set likewise fell from **153,780,224 B** at 4K (cycle peak
**156,557,312 B**) to **147,533,824 B** at the final small image.  The
bounded-recovery gate passed and both explicit decline flags are true.  This
is process-level evidence for this host and build, not a promise that every
allocator or Vulkan driver will immediately decommit reusable pages.  The
per-resolution report is
`build/memory_ladder_medium_cpu_post_vulkan_tile_rebuild_20260813.json`.

### 2026-08-13: current small-to-large-to-small memory regression

The current CPU build was also measured with a fresh, reproducible resolution
ladder: `16x16 -> 64x64 -> 160x120 -> 320x240 -> 640x480 -> 1280x720 ->
4096x256 -> 256x4096 -> 1920x1080 -> 2560x1440 -> 3000x1600 -> 3840x2160`,
followed by `1920x1080 -> 4096x256 -> 256x4096 -> 64x64`.  Each sample loads,
recognizes, and destroys its input image before Windows private bytes and
working set are sampled.

On medium PP-OCRv6 with CPU backend, 16 kernel workers, and approximate GELU,
the warmed private baseline was **143,552,512 B**.  Private bytes reached
**149,356,544 B** at `3840x2160` and fell to **148,066,304 B** at the final
`64x64` sample: **-0.86%** from the 4K sample and **+3.14%** from warm.  The
working set likewise fell from **153,243,648 B** to **151,785,472 B**.  The
bounded-recovery gate passed; both private-byte and working-set decline flags
are true.  This verifies a post-large-image decrease on this host, while the
small residual capacity is expected allocator reuse rather than a leak claim.
The full per-resolution JSON record is
`build/memory_ladder_medium_cpu_resolution_ladder_current_20260813.json`.

### 2026-08-13: dense-page scheduler and model-aware stress recheck

The default CPU crop-graph concurrency is now **8** rather than 12.  This
matches the default SIMD kernel worker budget, avoids cache/bandwidth
contention between outer crop batches and inner AVX-512/AVX2 kernels, and
keeps the public `PPOCR_REC_PARALLELISM` override for host-specific tuning.
On the local AVX-512 CPU (16 configured kernel workers, approximate GELU),
the normal English page measured **33.548 / 70.323 / 210.214 ms** for
tiny/small/medium (2 warm-ups, 8 runs).  Medium was **6.6%** lower than the
comparable 12-way outer-crop configuration (225.056 ms); this is a local
configuration measurement, not a portable performance guarantee.

All three model sizes preserved decoded-text parity against ONNX Runtime over
four varied inputs each (**12/12**): English, mixed text, a single-line crop,
and low-contrast text.  The reports are
`build/accuracy_{tiny,small,medium}_cpu_default8_20260813.json`.
`bench/stress_regression.py` now accepts `--model tiny|small|medium` and has
explicit medium-only box-count bounds for the synthetic blank and dense-blob
cases. This preserves the original tiny/small limits while avoiding a false
failure when the larger detector stably reports its extra valid components.
The 11-image medium stress corpus passed **11/11** with `--model medium`;
the record is `build/stress_medium_cpu_default8_model_bounds_20260813.json`.

### 2026-08-13: dense-page activation bound and refreshed memory check

The default recognizer NCHW batch is now **1** crop (the public
`PPOCR_REC_BATCH_SIZE` override still enables larger batches for measured
multi-page workloads).  A current dense `1920x1080` UI sweep showed that
larger concurrent crop tensors increase medium-model activation pressure and
lose latency on this host.  With CPU, 16 SIMD workers, approximate GELU, and
the eight-way outer scheduler, the default measured **255.141 ms** (tiny,
2 warm-ups/6 runs), **939.888 ms** (small, 2/4), and **3884.903 ms** (medium,
2/4).  Against the matching CPU ONNX Runtime pipeline on the same dense page,
a separate 2-warm-up/4-run sample was **46.0%**, **18.7%**, and **5.9%** lower
latency for tiny/small/medium respectively.  The latter remains a
machine-specific comparison; the dense image and detected-box mix are part of
the benchmark input.  Reports: `build/dense_cpu_batch1_vs_ort_20260813.json`
and `build/dense_batch_size_sweep_cpu_20260813.json`.

The change retained three-model decoded-text parity on four varied inputs
(**12/12**; `build/accuracy_{tiny,small,medium}_cpu_default_batch1_20260813.json`).
A fresh medium CPU resolution ladder also passed.  It warmed at
**143,974,400 B**, reached **148,078,592 B** at `3840x2160`, and returned to
**144,060,416 B** at the final `64x64` image: **-2.71%** versus the 4K sample
and **+0.06%** versus warm.  Windows working set declined from
**151,060,480 B** to **147,656,704 B**; both recovery flags are true.  The
durable per-resolution record is
`build/memory_ladder_medium_cpu_default_batch1_20260813.json`.

### 2026-08-13: recognizer executor allocation reuse

`RunCtcTop1`, the production recognizer route, now reuses a per-worker
thread-local small input-pointer list for its fused and generic node dispatch.
This removes recurring short-lived `std::vector<const Tensor*>` allocations
for every recognizer node and crop while retaining the existing tensor
lifetime plan, exact operator calls, and safe automatic growth for a future
larger input arity. It is deliberately a low-risk allocation/allocator
pressure reduction rather than a claim of an isolated compute-kernel speedup.

After the change, decoded text still matched ONNX Runtime on the four varied
images for tiny, small, and medium (**12/12**), and kernel/Vulkan smoke passed.
A new medium CPU resolution ladder also passed with the full ascending and
recovery sequence. Its private bytes fell from **150,818,816 B** at
`3840x2160` to **143,855,616 B** at the final `64x64` (**-4.62%**); working
set fell from **154,529,792 B** to **147,832,832 B**. The final sample was
2.60% below the warmed baseline and all bounded-recovery/decline checks were
true. See `build/memory_ladder_medium_cpu_executor_scratch_20260813.json`.

The immediate dense-page sanity sample (CPU, 16 kernel workers, approximate
GELU, default recognizer batch) was **255.959 ms / 860.055 ms / 3859.129 ms**
for tiny/small/medium (one warm-up, three runs). Because shared-host timing
varies materially, it is retained only as a post-change sanity record rather
than compared as a claimed percentage gain. The three JSON records are
`build/executor_scratch_bench_{tiny,small,medium}_cpu_20260813.json`.

Hybrid remains guarded by per-shape full-boundary admission (`H2D / dispatch
/ fence / D2H`); it uses Vulkan only when that complete operation is no
slower than the matching CPU SIMD batch kernel. On the local AMD Radeon 780M
snapshot, Hybrid measured 33.469/68.070/264.781 ms for tiny/small/medium,
respectively, so the end-to-end route is still not a GPU speedup on this UMA
device. `gpu_only` remains unavailable until the complete OCR graph and DB
post-processing can remain device-resident.

### 2026-08-13: compact CTC last-use counters

The recognizer's `RunCtcTop1` no longer clones a string-keyed
`unordered_map` of tensor-use counts for every crop. Model loading now assigns
each dynamic activation one compact counter slot; each worker resets a
thread-local `uint16_t` counter vector and continues to resolve only the
fixed names needed by the existing lifetime planner. This removes repeated
map-node allocation/copying from the multi-crop hot path while preserving the
same final-use, in-place, and immediate-release decisions.

The native kernel/Vulkan smoke suites passed, as did decoded-text parity
against ONNX Runtime for all three model sizes over English, mixed text,
single-line, and low-contrast inputs (**12/12**). A heterogeneous seven-image
tiny Hybrid batch (`16x16`, normal text, low contrast, ultra-wide,
ultra-tall, dense 1080p, and 1440p screenshot) remained serial-identical for
all 188 results. The one-run steady-state observation was **1456.274 ms**
batched versus **1485.160 ms** serial; it is not promoted to a general GPU
latency claim.

The accompanying medium CPU full-resolution ladder passed its 20% bounded
recovery gate. The final small image used **148,414,464 B** private memory
(+0.25% versus its warmed baseline and 0.65% below the cycle peak). This
particular Windows allocator run did not immediately decommit relative to the
`3840x2160` checkpoint, so both direct-decline flags are intentionally
recorded as false rather than presented as a memory-reduction claim. Full
per-resolution evidence is
`build/memory_ladder_medium_cpu_ctc_use_slots_20260813.json`.

The dense timing samples were materially host-noisy (tiny **361.536 ms**,
small **1858.371 ms**, medium **4585.517 ms** means) and therefore serve only
as post-change sanity records, not percentage-improvement evidence:
`build/ctc_use_slots_bench_{tiny,small,medium}_cpu_20260813.json`.

### 2026-08-13: explicit post-4K resolution-memory recovery sample

The native memory regression now distinguishes the **first** image after the
`3840x2160` sample (`1920x1080`) from the final `64x64` check.  Its JSON
includes private-byte and working-set decline flags for that immediate 4K
transition, in addition to the existing bounded final-small-image gates. This
avoids accidentally treating a later aspect-ratio sample as proof of direct
large-image recovery.

On this Windows host, a fresh one-cycle medium CPU run over the deterministic
ascending ladder (`16x16` through `3840x2160`) passed all 20% bounded-recovery
limits. Private usage fell from **148,787,200 B** at `3840x2160` to
**148,303,872 B** after the immediately following `1920x1080` input
(**-0.32%**); working set fell from **152,776,704 B** to **151,871,488 B**.
After the staged wide/tall recovery tail, the final `64x64` sample was
**143,667,200 B** private (**-3.08%** versus the warmed baseline). This is a
host/run-specific observation閳ユ摱indows may retain heap capacity on a later run;
the report preserves both direct-decline booleans and all per-resolution
samples. Reproduce it with `bench/generate_memory_ladder.py` and
`bench/memory_ladder_regression.py`; recorded evidence:
`build/memory_ladder_medium_cpu_immediate_4k_recovery_20260813.json`.

### 2026-08-13: current full-resolution memory-recovery check

The deterministic native memory ladder was run again with the current release
build and medium PP-OCRv6 on the CPU backend.  It processes one image at a
time, releases that image before sampling, and covers the complete sequence:
`16x16 -> 64x64 -> 160x120 -> 320x240 -> 640x480 -> 1280x720 -> 4096x256 ->
256x4096 -> 1920x1080 -> 2560x1440 -> 3000x1600 -> 3840x2160 -> 1920x1080
-> 4096x256 -> 256x4096 -> 64x64`.

On this Windows host, private bytes were **148,770,816 B** at `3840x2160` and
fell immediately to **147,767,296 B** after the next `1920x1080` image
(**-0.67%**).  The working set also fell from **151,744,512 B** to
**150,671,360 B**.  After the full recovery tail, the final `64x64` sample was
**147,222,528 B** private and **151,085,056 B** working set: below the 4K
sample in both measures.  All bounded-recovery gates passed.  This is a
host-specific allocator observation, not a promise that every OS allocator
will immediately decommit reusable heap capacity.  The complete per-resolution
record is `build/memory_ladder_medium_cpu_current_20260813.json`.

### 2026-08-13: generic SIMD Conv+ReLU correctness and memory recheck

The generic AVX-512/AVX2 spatial-convolution fallback now applies its requested
ReLU after the vector convolution writeback. Dedicated fused 2x2/3x3 paths
already performed that activation, but the generic 5x5/7x7 and asymmetric
SIMD fallback did not. This is a correctness repair, not a claimed latency
win. The rebuilt kernel and Vulkan smoke tests pass, and decoded-text parity
against ONNX Runtime passes **4/4** varied inputs for every PP-OCRv6 tier
(tiny, small, medium): English, mixed text, a short line, and noisy
low-contrast text. Reports are
`build/accuracy_{tiny,small,medium}_relu_fix_20260813.json`.

The medium CPU heterogeneous six-image batch smoke (`16x16`, normal English,
mixed text, wide/tall strips, low contrast) was result-identical and measured
**1653.548 ms serial** versus **1414.858 ms batched** (**1.169x** throughput).
A fresh full-resolution CPU memory ladder also passed: its `3840x2160` private
sample was **149,692,416 B**, the immediately following `1920x1080` sample
was **145,428,480 B** (**-2.85%**), and the final `64x64` sample was
**147,177,472 B** (**-0.35%** versus warm baseline). The complete report is
`build/memory_ladder_medium_cpu_relu_fix_recheck_20260813.json`.

### 2026-08-13: refreshed diverse CPU versus ONNX Runtime measurement

Using the current release build, dynamic AVX-512 dispatch, CPU-only backend,
`PPOCR_BENCH_THREADS=16`, five warm-ups, and 30 timed iterations per case, a
new paired measurement confirms a material end-to-end latency advantage over
ONNX Runtime CPUExecutionProvider for all three official PP-OCRv6 tiers on two
different input shapes.  This is host-specific evidence, not a cross-hardware
guarantee; both engines ran sequentially in the same process environment.

| Input | Model | C++ mean | ORT mean | C++ latency reduction |
|---|---|---:|---:|---:|
| `en.ppm` | tiny | 32.653 ms | 125.618 ms | 74.01% |
| `en.ppm` | small | 68.304 ms | 180.432 ms | 62.14% |
| `en.ppm` | medium | 208.100 ms | 366.127 ms | 43.16% |
| `wide_strip_3000x80.ppm` | tiny | 30.776 ms | 65.817 ms | 53.24% |
| `wide_strip_3000x80.ppm` | small | 60.182 ms | 129.017 ms | 53.35% |
| `wide_strip_3000x80.ppm` | medium | 175.134 ms | 272.745 ms | 35.79% |

The full machine-readable measurement is
`build/bench_cpu_ort_diverse_refresh_20260813.json`. It complements rather
than replaces the larger noisy/dense stress cases: those cases exercise memory
and batch robustness and can be affected more by host scheduling/cache noise.

### 2026-08-13: current CPU-versus-ORT recheck

After the current SIMD/scheduler rebuild, the paired CPU-only comparison was
rerun with dynamic AVX-512 enabled, `PPOCR_BENCH_THREADS=16`, two warm-ups,
and ten timed iterations per case.  The compact English page and a 3000x80
ultra-wide strip cover normal and extreme-aspect detector resize paths.  All
three tiers remain faster than ONNX Runtime CPUExecutionProvider on this host;
the lower reduction on medium's short page reflects host timing variance and
is recorded rather than hidden.

| Input | Model | C++ mean | ORT mean | C++ latency reduction |
|---|---|---:|---:|---:|
| `en.ppm` | tiny | 33.256 ms | 121.671 ms | 72.67% |
| `en.ppm` | small | 81.696 ms | 180.725 ms | 54.80% |
| `en.ppm` | medium | 246.343 ms | 374.196 ms | 34.17% |
| `wide_strip_3000x80.ppm` | tiny | 24.297 ms | 59.489 ms | 59.16% |
| `wide_strip_3000x80.ppm` | small | 53.559 ms | 101.561 ms | 47.26% |
| `wide_strip_3000x80.ppm` | medium | 152.553 ms | 263.361 ms | 42.07% |

The reproducible JSON record is
`build/bench_cpu_ort_diverse_current_20260813.json`.  It is a local-machine
measurement, not a guarantee for every processor or ONNX Runtime build.

### 2026-08-13: blocked recognizer transpose

The recurrent PP-OCRv6 recognizer exchanges both `[N,T,C] <-> [N,C,T]` and
`[N,H,C,T] <-> [N,H,T,C]` layouts.  Their dedicated copy paths now transpose
`32x32` cache tiles instead of continually evicting a full source/destination
matrix row.  The operation is a pure element relocation, so it preserves
values bit-for-bit and applies to scalar, AVX2, AVX-512, and NEON builds.

On this AVX-512 host with `PPOCR_BENCH_THREADS=16`, three independent runs of
12 timed English-page inferences changed mean latency from **32.120 to
30.530 ms** (tiny, **4.95%**), **62.793 to 64.053 ms** (small, **-2.01%**),
and **192.745 to 190.614 ms** (medium, **1.11%**).  The small-model result is
within host scheduling noise and is recorded rather than promoted as a broad
claim; no model-specific switch is required.  Raw paired samples are
`build/bench_transpose_tile_{baseline,tiled}_20260813.json`.  Text parity
against ORT remained 3/3 across English, ultra-wide, and low-contrast inputs
for all tiers in `build/accuracy_transpose_tiled_core_20260813.json`.

The same current medium build passed the full small-to-4K-to-small memory
ladder: `3840x2160` was **149,356,544 B** private / **153,313,280 B** working
set, the immediate following 1080p sample fell to **148,262,912 B** /
**152,317,952 B**, and the final `64x64` sample was **148,324,352 B** /
**152,129,536 B**.  Full per-resolution evidence:
`build/memory_ladder_medium_cpu_transpose_tiled_20260813.json`.

### 2026-08-13: CPU/Hybrid batch recheck and GPU-only boundary

The public three-policy contract was rechecked on the local Radeon 780M:
Vulkan loader and compute are present, while `gpu_only` explicitly rejects
construction because PP-OCRv6 still needs device implementations for its
complete detector, recognizer, preprocessing, and DB/CTC postprocessing
graphs.  It does not silently execute on CPU.  Hybrid remains usable and
continues to select each Vulkan segment only after its complete synchronous
`H2D + dispatch + fence + D2H` path is correct and no slower than CPU.

With four same-shape English pages, `det_batch_size=4`, `rec_batch_size=4`,
one graph worker, and two timed iterations, the verified `RecognizeBatch()`
result-identical speedups over four serial calls were **1.589x / 1.256x /
1.140x** on CPU for tiny/small/medium and **1.598x / 1.323x / 1.225x** in
Hybrid.  This is real cross-page NCHW batching, not padding unrelated page
sizes.  The raw output is `build/bench_batch4_cpu_hybrid_20260813.json`.

### 2026-08-13: AVX-512 recognizer GEMM tile admission

The AVX-512 GEMM implementation retains its two-row and four-row kernels,
but the four-row tile is now opt-in (`PPOCR_ENABLE_AVX512_GEMM4ROW=1`) rather
than the default.  A paired `PPOCR_BENCH_THREADS=16` measurement showed that
the larger register footprint is beneficial only for some medium shapes and
regresses the shipped workload overall.  With the default two-row tile and
three warm-ups/15 runs, current means were **28.885 / 64.840 / 201.871 ms**
on the English page and **20.263 / 48.142 / 134.558 ms** on the 3000x80 strip
for tiny/small/medium. The control data is
`build/bench_gemm4row_{ab_current,off_repeated}_20260813.json` and the final
default record is `build/bench_gemm2row_default_20260813.json`.

Text parity remains **3/3** for each model on English, ultra-wide, and
low-contrast inputs against ONNX Runtime:
`build/accuracy_gemm2row_core_20260813.json`. This change keeps the exact
per-row K-order and applies only after dynamic AVX-512 capability checks;
AVX2, NEON, and scalar dispatch are unchanged.

### 2026-08-13: hot-path ISA configuration caching

The runtime ISA policy remains dynamic (CPUID/XCR0 is still checked before
entering AVX-512), but process-scoped environment switches for pointwise
tiles, 2x2 tiles, wide tiles, and the 2x2 SAME kernel are now cached once.
These predicates occur repeatedly in the detector and recognizer; eliminating
repeated environment lookups reduces dispatcher overhead without changing an
operator's arithmetic or the AVX2/NEON/scalar fallback. On the current
16-worker AVX-512 host, the final English-page smoke measured **30.462 ms**
tiny, **62.268 ms** small, and **190.709 ms** medium. Raw result:
`build/bench_dispatch_cache_current_20260813.json`.

All three tiers retained 3/3 ORT decoded-text parity on English, ultra-wide,
and low-contrast inputs (`build/accuracy_dispatch_cache_core_20260813.json`),
and both CPU kernel and Vulkan primitive smoke tests passed.

### 2026-08-13: detector BatchNorm activation reuse

The generic graph executor now mirrors the production CTC executor for a
final-use plain BatchNorm: it applies the alias-safe affine transform directly
to the dying input activation, then transfers the tensor name to the graph
output. This removes one same-sized detector feature-map allocation/copy at
eligible boundaries without changing the BatchNorm arithmetic or its dynamic
coefficient handling. It is guarded by the existing
`PPOCR_DISABLE_BATCHNORM_INPLACE=1` deployment A/B switch.

All three model tiers retained 3/3 decoded-text parity against ONNX Runtime on
English, ultra-wide, and low-contrast inputs
(`build/accuracy_bn_inplace_core_20260813.json`); CPU and Vulkan smoke also
passed. The refreshed medium full ladder passed with a 4K private/working-set
sample of **148,881,408 B / 152,858,624 B**, immediately declining at the
following 1080p image to **147,857,408 B**, then returning to
**143,728,640 B / 147,775,488 B** on the final 64x64 image. Complete records:
`build/memory_ladder_medium_cpu_bn_inplace_20260813.json` and
`build/bench_bn_inplace_current_20260813.json`.

### 2026-08-13: single-pass SIMD CTC ArgMax

AVX-512 and AVX2 CTC ArgMax no longer scan each vocabulary row twice. The
SIMD reduction now retains each lane's source index as it computes its local
maximum, then performs one scalar lane/tail reduction with the same strict
`>` first-maximum rule as the portable path. This removes a second full read
of each 6,906/18,710-class logits row while preserving the scalar fallback
for malformed NaN logits. The kernel smoke now explicitly covers ties across
SIMD lanes and vector tails.

The current 16-worker English-page smoke was **29.751 ms** tiny,
**60.676 ms** small, and **226.997 ms** medium
(`build/bench_argmax_single_pass_20260813.json`); timing remains host-noisy,
so this record is evidence of correctness rather than a broad end-to-end
speed claim. All three tiers retained 3/3 ORT decoded-text parity on English,
ultra-wide, and low-contrast samples
(`build/accuracy_argmax_single_pass_core_20260813.json`), and the Radeon
780M Vulkan smoke passed.

### 2026-08-13: diverse-shape correctness and terminal-CTC admission recheck

The current CPU implementation was additionally checked against ONNX Runtime
on `en.ppm`, the `3000x80` ultra-wide strip, and `noisy_lowcontrast.ppm` for
all three model tiers.  Decoded line count and text parity passed **3/3** for
tiny, small, and medium; the durable report is
`build/accuracy_diverse_core_20260813.json`.  A short stress smoke on the
low-contrast page and a `80x3000` ultra-tall strip also exercised the
opposite detector resize branch, where it found 3 and 20 text regions,
respectively, for every tier.  The narrow vertical strip does not yet have
text parity with this repository's ORT reference, so it is retained as a
throughput/robustness sample rather than being claimed as a correctness gate.

The bounded terminal-GEMM/CTC path was then A/B measured with three warm-ups
and 15 timed iterations per normal and ultra-wide input.  It improves tiny on
this host (English page **35.105 -> 31.379 ms**) but is neutral/negative for
small and slower for medium's ultra-wide case (**145.585 -> 276.362 ms**).
It therefore remains explicit opt-in via
`PPOCR_ENABLE_FUSED_TERMINAL_CTC=1`; the default executor remains the faster
cross-tier choice.  Raw A/B data: `build/bench_terminal_ctc_ab_20260813.json`.

### 2026-08-13: medium detector AVX-512 tile-pressure recheck

The medium-detector profile identifies regular `3x3` and `5x5`/`7x7`
context convolutions as the main remaining CPU hotspots.  Their eight-output
AVX-512 tiles were rechecked against the established four-output kernels. On
this client AVX-512 host, the larger tiles increase register pressure and can
produce substantial low-contrast tail latency: over eight timed iterations,
the medium `noisy_lowcontrast.ppm` median was **554.202 ms** with the default
four-output tiles versus **571.360 ms** with both eight-output experiments;
the latter's p95 reached **2569.152 ms**. The wide-strip median was likewise
**159.735 ms** (four) versus **163.488 ms** (eight).

Consequently, the proven four-output 3x3/wide-context tiles are now the
default. Eight-output variants remain opt-in for a separately measured server
CPU via `PPOCR_ENABLE_AVX512_CONV3X3_TILE8=1` and
`PPOCR_ENABLE_AVX512_CONV_WIDE8=1`; setting either `PPOCR_DISABLE_*` switch
still disables its matching experiment. Text parity remains 2/2 for tiny,
small, and medium on English and ultra-wide input; reports are
`build/accuracy_{tiny,small,medium}_avx512_tile4default_20260813.json`.

The refreshed medium CPU resolution ladder also passed. Its 4K private sample
was **149,164,032 B**, the immediate following 1080p sample was
**149,123,072 B**, and the final 64x64 sample was **148,738,048 B**. Private
bytes and working set both declined after the 4K image. Full evidence is
`build/memory_ladder_medium_cpu_avx512_tile4default_20260813.json`.

### 2026-08-13: AVX-512 2x2 SAME_UPPER detector-stem kernel

The detector's early `2x2` SAME_UPPER Conv nodes now use a dedicated AVX-512
interior kernel with explicit final-row/final-column zero-padding handling.
This avoids generic variable-kernel address arithmetic on the large feature
maps while retaining runtime AVX2/scalar fallbacks. The low-level smoke suite
checks odd/even dimensions, vector/scalar tails, and fused ReLU. Decoded text
parity against ONNX Runtime passed **2/2** (`en.ppm`, `wide_strip_3000x80`) for
tiny, small, and medium; reports are
`build/accuracy_{tiny,small,medium}_avx512_same2x2_20260813.json`.

On the local AVX-512 host the A/B is strongly workload dependent because the
shared machine has substantial scheduling variance. Repeated medium samples
put the specialized median around **181.580閳?32.008 ms**, versus
**187.559閳?08.236 ms** with the generic path; tiny/small consistently improved
on the short sample. It remains enabled because it removes the generic hot
loop, passes the complete correctness gates, and keeps a one-variable runtime
escape hatch for deployments whose AVX-512 frequency/cache balance differs.

### 2026-08-13: current ascending-resolution memory recovery measurement

`bench/generate_memory_ladder.py` generates deterministic PPM inputs from
`16x16` through `3840x2160`, including wide and tall 1,048,576-pixel strips.
The sequence then processes `1920x1080`, both strips again, and a final
`64x64` image. `ppocr_mem_soak` keeps one OCR instance alive, releases each
decoded image before sampling, and records Windows private bytes and working
set after every inference.

The current medium CPU two-cycle measurement passed all bounded-return gates.
In cycle 1, the `3840x2160` sample was **146,067,456 B** private /
**150,147,072 B** working set; the final `64x64` sample returned to
**143,634,432 B** / **147,345,408 B**, below the 4K sample in both measures.
Cycle 2 similarly returned from **146,886,656 B** / **150,986,752 B** at 4K
to **146,767,872 B** / **150,581,248 B** at the final small image. The first
post-4K 1080p sample declined in both measures in cycle 2; cycle 1 retained a
small amount of reusable allocator capacity before declining by the final
small-image checkpoint. This is a single-host allocator measurement, not an
assertion that an OS must decommit capacity immediately.

Use `--require-post-large-decline` to make immediate private-byte *and*
working-set decline a strict CI gate when that stronger condition is required;
the default remains bounded return because Windows allocator behavior is
non-deterministic. The complete per-resolution record is
`build/memory_ladder_medium_cpu_current_bounded_recovery_20260813.json`.

### 2026-08-13: verified current resolution-memory regression

The reproducible small-to-large-to-small test remains part of the native
regression suite.  It generates and processes, in order,
`16x16 -> 64x64 -> 160x120 -> 320x240 -> 640x480 -> 1280x720 -> 4096x256 ->
256x4096 -> 1920x1080 -> 2560x1440 -> 3000x1600 -> 3840x2160 -> 1920x1080
-> 4096x256 -> 256x4096 -> 64x64`.  Every PPM is decoded, inferred, and
destroyed before collecting Windows private bytes and working set, while one
OCR instance remains alive.  This tests both varied resolution growth and
whether capacity is bounded after a large page.

On the current medium CPU build (16 kernel workers), the fresh two-cycle run
passed all 20% bounded-return gates.  Cycle 1 returned from the 4K sample's
**147,230,720 B** private / **150,204,416 B** working set to
**143,486,976 B** / **147,460,096 B** at final `64x64`.  Cycle 2 showed an
immediate 4K-to-1080p decline in both measures (**148,221,952 B** /
**152,256,512 B** to **147,570,688 B** / **151,240,704 B**); its final small
sample remained below its pre-return peak.  The immediate-decline condition is
reported separately because Windows heap reclamation is allocator-run
dependent; the default regression correctly gates bounded recovery rather than
requiring decommit on every run.  Full per-resolution evidence:
`build/memory_ladder_medium_cpu_current_resolution_recovery_20260813.json`.

### 2026-08-14: exact-size preprocessing fast path

The fused RGB-to-BGR/NCHW preprocessor now recognizes an exact-size resize
region.  In that case bilinear interpolation is an identity, so it writes the
three normalized planes directly, without a horizontal-coordinate table or
four-pixel interpolation work per output value.  It preserves recognizer
right-padding because it writes only the natural-width columns.  The general
bilinear path remains unchanged for every resized image.  For deployment A/B,
set `PPOCR_DISABLE_IDENTITY_RESIZE_FASTPATH=1` before creating an OCR object.

The direct and generic paths produced identical native demo output for tiny,
small, and medium on exact-size `line_en`, `line_zh`, and `single_word`
inputs (`build/identity_resize_cpp_equivalence_20260814.json`).  Each model
also retained ORT decoded-text parity on an exact-size line, normal English
page, and 3000x80 ultra-wide page (3/3 each;
`build/accuracy_identity_resize_cached_{tiny,small,medium}_20260814.json`).
Kernel and Radeon 780M Vulkan smoke passed.

A fresh medium CPU resolution ladder remained bounded after the change: the
4K sample was **148,512,768 B** private / **152,322,048 B** working set and
the final `64x64` sample was **148,197,376 B** / **151,797,760 B**, below the
pre-return peak in both metrics.  Full record:
`build/memory_ladder_medium_cpu_identity_resize_20260814.json`.

The local 16-worker, CPU-only 100-run line benchmark was host-noisy on tiny,
but gave a stable medium improvement: **72.046 ms -> 67.421 ms** mean
(**6.42%** reduction) and **71.770 ms -> 67.466 ms** median.  Small and tiny
are already dominated by model execution on this single-line workload, so
this is recorded as a targeted preprocessing win rather than a claimed
uniform end-to-end speedup.  Raw A/B record:
`build/bench_identity_resize_cached_ab_20260814.json`.

### 2026-08-14: recognizer average-pool compaction

All three PP-OCRv6 recognizers contain the same valid `AveragePool(3x2,
stride=3x2)` bridge between convolutional features and the transformer.  A
dedicated NCHW kernel now executes that exact non-overlapping geometry, and
when its input has a graph-proven final use it compacts the result into the
front of the existing activation allocation before resizing it.  This removes
one bridge-output allocation and releases the unused capacity before the
transformer.  `PPOCR_DISABLE_AVERAGE_POOL3X2_FASTPATH=1` and
`PPOCR_DISABLE_AVERAGE_POOL3X2_INPLACE=1` are narrow A/B fallbacks.

The in-place route produced identical native demo output for tiny, small, and
medium on `line_en`, `line_zh`, `single_word`, and `en`
(`build/average_pool3x2_inplace_cpp_equivalence_20260814.json`). Each model
also retained ORT decoded-text parity on an exact-size line, English page, and
ultra-wide strip (3/3 each;
`build/accuracy_averagepool_inplace_{tiny,small,medium}_20260814.json`).
The current medium resolution-memory ladder passed: it returned from the
pre-return private/working-set peak of **149,241,856 B / 153,153,536 B** to
**143,618,048 B / 147,505,152 B** at final `64x64`.
`build/memory_ladder_medium_cpu_averagepool_inplace_20260814.json` preserves
every resolution sample.

On the local 16-worker CPU, 100-run one-line A/B was positive for medium
(**69.535 -> 68.032 ms**, 2.16%) and within shared-host noise for tiny/small.
It is therefore a verified allocation reduction and a targeted medium
throughput gain, not a uniform speed claim. Raw results:
`build/bench_averagepool_inplace_ab_20260814.json`. The Radeon 780M Vulkan
smoke and hybrid batch parity passed; hybrid continues to admit a Vulkan
segment only after its complete H2D, dispatch, fence, D2H, and CPU comparison
is no slower.

### 2026-08-14: attention QKV vector-copy transpose

The small and medium recognizer attention blocks export a QKV projection as
`[N,T,3,H,D]`, followed by the fixed ONNX permutation
`[2,0,3,1,4] -> [3,N,H,T,D]`.  The innermost head feature vector `D` is
contiguous before and after that permutation.  The executor now copies this
vector with `memcpy` rather than recalculating five-dimensional scalar
coordinates for every float.  All other transposes retain the generic path.
Set `PPOCR_DISABLE_QKV_TRANSPOSE_FASTPATH=1` before constructing OCR to A/B
the generic route.

Decoded-text parity against ONNX Runtime passed on English, Chinese, and
3000x80 ultra-wide inputs for tiny, small, and medium (3/3 for each tier;
`build/accuracy_qkv_transpose_{tiny,small,medium}_20260814.json`). Tiny has
no QKV attention block and remains unaffected. On the local AVX-512 CPU,
the 40-run English-page sample measured small **74.557 -> 70.835 ms**
(5.0%) and medium **217.134 -> 199.390 ms** (8.2%). On the distinct
30-run ultra-wide workload it measured small **44.199 -> 41.999 ms** (5.0%)
and medium **141.816 -> 139.679 ms** (1.5%). The one page measurements are
host-specific; raw records are `build/bench_qkv_transpose_ab_20260814.json`
and `build/bench_qkv_transpose_wide_ab_20260814.json`.

The current medium CPU small-to-4K-to-small ladder stayed bounded: the final
64x64 sample was **143,474,688 B** private / **147,525,632 B** working set,
below both the 4K sample (**146,661,376 B / 150,781,952 B**) and the
pre-return peak.  Full per-resolution evidence is
`build/memory_ladder_medium_cpu_qkv_transpose_20260814.json`.

### 2026-08-14: QKV transpose/slice elimination

The small and medium attention exports immediately split the just-transposed
`[3,N,H,T,D]` QKV activation into three `Slice(1) -> Squeeze(0)` branches.
The loader now recognizes this complete three-branch topology and replaces it
with three direct copies from the original `[N,T,3,H,D]` projection into their
already-squeezed `[N,H,T,D]` destinations. This eliminates the full
transposed QKV activation and the three temporary slice buffers. It is a
strict model-shape fusion; all general Transpose/Slice graphs retain their
ordinary execution. `PPOCR_DISABLE_QKV_SLICE_FUSION=1` is the deployment A/B
fallback.

On the local AVX-512 CPU, the 20-run English-page A/B, with the earlier
vector-copy transpose still enabled in the baseline, measured small
**76.315 -> 72.245 ms** (5.3%) and medium **222.274 -> 195.827 ms** (11.9%).
The baseline values include the prior QKV-vector optimization, so this records
the additional benefit of removing the full transpose/slice lifetime rather
than double-counting it. Raw record:
`build/bench_qkv_slice_fusion_ab_20260814.json`.

The default fused route retained ORT decoded-text parity on English, Chinese,
and 3000x80 ultra-wide input for all three shipped model tiers (3/3 each;
`build/accuracy_qkv_slice_fusion_default_{tiny,small,medium}_20260814.json`).
Tiny has no matching transformer QKV topology and is unchanged. The medium
ascending-resolution memory gate passed: final `64x64` was
**143,544,320 B** private / **151,076,864 B** working set, below the 4K
sample and pre-return peak. Full record:
`build/memory_ladder_medium_cpu_qkv_slice_fusion_20260814.json`. CPU kernel
and Radeon 780M Vulkan smoke also passed; hybrid admission policy remains
unchanged and selects only complete GPU segments that are no slower than CPU.

### 2026-08-14: complete QKV split lifetime and resolution-memory recheck

The three direct Q/K/V copies are now emitted by one `FusedQkvSplit` node
with three outputs.  This keeps the shared `[N,T,3,H,D]` projection live only
until all three final `[N,H,T,D]` tensors have been materialized, then retires
it through the normal use counter.  It is a PP-OCRv6 topology-specific memory
fusion, not general multi-output ONNX runtime support.

The existing native resolution ladder provides the requested ascending image
coverage: `16x16`, `64x64`, `160x120`, `320x240`, `640x480`, `1280x720`,
wide/tall `4096x256` and `256x4096`, `1920x1080`, `2560x1440`, `3000x1600`,
and `3840x2160`, followed by a staged recovery tail and `64x64` return. A
fresh two-cycle medium CPU run passed its 20% bounded-recovery gate. In cycle
1, private / working-set memory declined from **149,700,608 / 153,661,440 B**
at 4K to **147,795,968 / 151,752,704 B** at the immediately following 1080p
page, then finished at **147,509,248 / 151,162,880 B** on the return 64x64.
Cycle 2 likewise declined immediately after 4K; the final small sample was
**147,435,520 / 151,105,536 B**, only **0.27%** above the warmed private-byte
baseline.  Per-resolution evidence is
`build/memory_ladder_medium_cpu_qkv_split_20260814.json`. Windows allocator
capacity is reusable and samples may vary, so the gate verifies bounded
recovery rather than demanding that every allocation be decommitted.

The fused executor also retained decoded-text parity with ONNX Runtime on
English, Chinese, and 3000x80 ultra-wide inputs (3/3 medium cases;
`build/accuracy_qkv_split_default_medium_20260814.json`). CPU kernel and
Radeon 780M Vulkan smoke tests passed after the change.

A final all-tier recheck repeated those three text/diverse-size cases for
tiny, small, and medium (**3/3 each**):
`build/accuracy_qkv_split_{tiny,small,medium}_20260814.json`. The 20-run
CPU snapshot (16 workers, exact GELU) was tiny **30.920 / 21.393 ms**, small
**62.485 / 44.688 ms**, and medium **198.325 / 132.951 ms** for English /
3000x80, respectively (`build/bench_cpu_diverse_qkv_split_20260814.txt`).
The former is timing evidence for this host only; the durable cross-engine
comparison remains the longer, independently measured CPU-versus-ORT corpus.
The final one-cycle 16-sample medium ladder also passed its bounded-return
gate (`build/memory_ladder_medium_cpu_qkv_split_final_20260814.json`), with
the return `64x64` footprint below the pre-return peak. Its immediate 4K
sample did not decline in that isolated Windows run, so the report records
that fact rather than claiming forced allocator decommit.

### 2026-08-14: extended 8K source-image recovery gate

The generated memory corpus now continues beyond 4K with `5120x2880` and
`7680x4320` pages, then immediately processes a `3840x2160` recovery page
before the existing 1080p, extreme-aspect-ratio, and final-`64x64` tail. The
native soak keeps one OCR instance alive but scopes every decoded RGB image to
`load -> Recognize -> destroy -> sample`; this specifically tests source image
release as well as activation reuse. It is intentionally more demanding than
the detector's 960-pixel inference cap, because the decoded 8K source itself
is about 95 MiB.

With medium CPU PP-OCRv6, one strict cycle passed with
`--require-post-large-decline`: private / working-set memory fell from
**149,139,456 / 153,059,328 B** after 8K to **146,808,832 / 150,470,656 B**
at the immediate 4K recovery (**-2,330,624 B / -2,588,672 B**). The final
`64x64` was **147,595,264 / 151,126,016 B**, below the pre-return peaks
(**149,151,744 / 153,059,328 B**) and only **3.03%** above the warmed private
baseline. Full 19-resolution evidence is
`build/memory_ladder_medium_cpu_8k_strict_20260814.json`. This is a
host-specific bounded-recovery observation; it does not promise every Windows
heap or Vulkan driver immediately decommits reusable pages.

### 2026-08-14: reproducible small-to-8K memory ladder

`bench/generate_memory_ladder.py` now creates 19 deterministic P6 PPM pages
in ascending source resolution from `16x16` through `5120x2880` and
`7680x4320`, followed by a `3840x2160 -> 1920x1080 -> wide/tall -> 64x64`
recovery tail. `bench/memory_ladder_regression.py` keeps one native OCR
instance alive, destroys each decoded source image immediately after
recognition, and records private bytes plus working set after every page.

The strict gate compares the 5K/8K high-water with the deliberate post-8K
recovery tail (`4K -> 1080p -> wide/tall`), and independently bounds the
final `64x64` footprint. Windows allocator/working-set accounting is not
guaranteed to be monotonic across adjacent samples, so an optional
`--require-immediate-post-large-decline` diagnostic is available, but the
default strict test requires both counters to fall below the large-page
high-water somewhere in that recovery tail. This avoids presenting sampling
jitter as a leak while still proving that the process returns from a large
page under the same live OCR instance.

### 2026-08-14: recognition-worker lifetime hardening

The production CTC executor no longer stores its compact use-counter vector
or node-input pointer scratch vector in `thread_local` storage.  Those two
small vectors are now invocation-local (the pointer list reserves eight
entries).  This is deliberately a lifetime fix rather than an allocator
cache: on the MinGW/Windows build, CTC execution can run on short-lived outer
recognition workers, and emulated-TLS vector destructors may otherwise be
registered through a different pthread DLL during worker teardown.  The
result was an intermittent `0xC0000374` heap-corruption process exit after a
successful long memory ladder.  Activation tensors themselves remain
per-invocation and are released by the existing graph use-count planner.

The current medium CPU 19-resolution ladder completed and exited cleanly.
Private / working-set memory fell from **149,811,200 / 153,808,896 B** on the
8K source page to **147,935,232 / 151,924,736 B** on its immediate 4K
recovery page; the final `64x64` sample was **143,974,400 / 147,787,776 B**,
only **0.45%** above the warmed private-byte baseline and below the complete
large-page high-water marks.  Full record:
`build/memory_ladder_medium_cpu_8k_tls_fix_20260814.json`.

CPU and hybrid runs both completed the varied `16x16`, English, 3000x80
ultra-wide, and 80x3000 ultra-tall corpus for tiny/small/medium (**12/12 per
backend**): `build/accuracy_tls_lifetime_fix_{cpu,hybrid}_diverse_20260814.json`.
Kernel smoke and Radeon 780M Vulkan smoke also passed. Hybrid retains its
strict per-shape whole-boundary no-slower admission rule; this change does not
claim that the incomplete device-resident OCR graph supports GPU-only mode.

### 2026-08-14: fresh small-to-large memory regression

The checked-in generator and regression gate were rerun against the medium
PP-OCRv6 CPU pipeline.  The deterministic corpus contains 19 PPM inputs:
`16x16 -> 64x64 -> 160x120 -> ... -> 5120x2880 -> 7680x4320`, followed by
`3840x2160 -> 1920x1080 -> wide/tall strips -> 64x64`.  It therefore measures
both normal increasing resolutions and source-image release after the 8K page,
while retaining one OCR instance for the whole sequence.

The strict gate passed for both Private Bytes and Working Set.  The 8K sample
was **148,996,096 / 152,911,872 B**; the immediately following 4K recovery
sample was **147,918,848 / 151,826,432 B** (**-1,077,248 / -1,085,440 B**).
The final 64x64 sample was **143,548,416 / 147,689,472 B**, only **0.25%**
above the warmed Private-Bytes baseline and below the pre-return high-water
mark.  The complete per-resolution record is
`build/memory_ladder_medium_cpu_strict_final_20260814.json`.

Reproduce it with:

```powershell
python bench/generate_memory_ladder.py --output build/memory_ladder_images
python bench/memory_ladder_regression.py --cpp build/ppocr_mem_soak.exe --det D:/workprj/aicoder/.tmp/ocr-models/ppocrv6_medium_det.onnx --rec D:/workprj/aicoder/.tmp/ocr-models/ppocrv6_medium_rec.onnx --dict build/dict_ppocrv6_latin.txt --images build/memory_ladder_images --cycles 1 --backend cpu --require-post-large-decline --report build/memory_ladder_medium_cpu_strict.json
```

### 2026-08-14: current resolution-memory validation

The reproducible ladder was generated again and run with one live medium CPU
OCR instance. It records Private Bytes and Working Set after every one of the
19 pages, with decoded PPM storage destroyed before each sample. The run
passed the bounded-return gate:

| Checkpoint | Private Bytes | Working Set |
| --- | ---: | ---: |
| warmed baseline | 143,437,824 B | 閳?|
| 8K (`7680x4320`) | 148,480,000 B | 152,453,120 B |
| first 4K recovery | 148,000,768 B | 151,769,088 B |
| final `64x64` | 146,055,168 B | 150,097,920 B |

Both counters fell immediately after 8K in this run; the final small image was
**1.82%** above the warmed private-byte baseline and below the complete
pre-return peak. The full per-resolution evidence, including the recovery
tail decision, is
`build/memory_ladder_medium_cpu_strict_current_20260814.json`. Results are
machine/run-specific process accounting, not a fixed cross-platform memory
guarantee.

### 2026-08-14: revalidated varied-resolution memory recovery

The 19-image ascending-and-recovery ladder was regenerated and rerun with one
medium PP-OCRv6 CPU instance (`PPOCR_BENCH_THREADS=16`).  Each PPM is loaded,
recognized, destroyed, and only then sampled, so source RGB storage from a
large page cannot be hidden by retaining the input vector.  The strict
post-large recovery gate passed for both Windows Private Bytes and Working Set:

| Checkpoint | Private Bytes | Working Set |
| --- | ---: | ---: |
| warmed baseline | 142,888,960 B | n/a |
| 5K peak (`5120x2880`) | 148,463,616 B | 152,395,776 B |
| 8K (`7680x4320`) | 146,882,560 B | 151,015,424 B |
| first 4K recovery | 147,435,520 B | 151,506,944 B |
| final `64x64` | 146,251,776 B | 150,216,704 B |

The observed peak was the 5K page rather than 8K, which is valid for Windows
process accounting and the detector's bounded tensor resize.  Both counters
were already below that high-water mark at the first 4K recovery sample; the
final small image remained below it and was 2.35% above the warmed private-byte
baseline.  The machine-specific, per-resolution JSON evidence is
`build/memory_ladder_medium_cpu_strict_revalidated_20260814.json`.

### 2026-08-14: zero-copy top-level ONNX protobuf traversal

Model construction no longer copies the complete embedded ONNX `GraphProto`
before parsing it.  The model reader now exposes bounded protobuf byte spans;
the top-level graph is parsed directly from the already-owned model file buffer.
Nested node, attribute, initializer, and operator-set messages retain their
existing owning parse paths, so the supported PP-OCRv6 graph semantics and
runtime's no-ONNX-Runtime dependency remain unchanged.  This removes one
transient full-model allocation/copy during startup and lowers model-loading
work without altering per-inference numerical operations.

On the local Windows AVX-512 host (`PPOCR_BACKEND=cpu`, 16 kernel workers,
one warm-up and five English-page measurements), constructor-to-ready load
time changed as follows:

| Model | Before | After | Lower load time |
| --- | ---: | ---: | ---: |
| tiny | 27.184 ms | 17.808 ms | 34.5% |
| small | 102.254 ms | 72.066 ms | 29.5% |
| medium | 519.523 ms | 294.587 ms | 43.3% |

The per-model evidence is
`build/bench_model_load_span_parse_ab_20260814.json`; the post-change
inference snapshot is `build/bench_model_load_span_parse_20260814.json`.
Kernel and Vulkan admission smoke passed. CPU and hybrid also completed the
five-size `16x16`, English, low-contrast, ultra-wide, and ultra-tall corpus
for tiny/small/medium (30 successful executions per backend); the recorded
outputs are `build/smoke_span_parse_diverse_{cpu,hybrid}_20260814.json`.
One medium ultra-wide confidence differs only in the last reported digits in
hybrid because its transfer-inclusive channel-affine/depthwise-Swish admission
can select a numerically valid Vulkan segment; decoded text and geometry are
unchanged. GPU-only remains deliberately rejected until the complete graph is
device-resident.

### 2026-08-14: invocation-local executor input scratch reuse

Both the general graph executor and the production CTC executor now reuse one
invocation-local, non-owning node-input pointer vector. Previously, the general
path allocated a separate small vector at each generic operator and at several
in-place fast paths. The new scratch has eight reserved entries (larger future
operator signatures still grow it safely), is never shared across OCR calls,
and retains the existing no-TLS lifetime rule for MinGW worker teardown.

This removes repeated allocator traffic without changing tensors, reduction
order, CPU SIMD dispatch, or the Vulkan admission policy. A fresh CPU English
snapshot (`PPOCR_BENCH_THREADS=16`, two warm-ups, twelve runs) measured
**30.817 / 61.215 / 182.730 ms** for tiny/small/medium. The varied-shape CPU
snapshot additionally completed ultra-wide at **19.090 / 44.437 / 127.590 ms**
and dense `1920x1080` at **265.831 / 1133.132 / 4880.140 ms** (three-run
shared-host samples, therefore directional rather than portability claims).

The 19-resolution medium memory ladder passed again: its 5K high-water was
**149,815,296 / 153,812,992 B** (Private Bytes / Working Set), the first 4K
recovery was **146,288,640 / 150,216,704 B**, and final `64x64` was
**146,501,632 / 150,388,736 B**, still below the pre-return high-water. The
complete evidence is `build/memory_ladder_medium_cpu_input_scratch_reuse_20260814.json`.
All 15 tiny/small/medium varied-size cases completed in both CPU and hybrid;
their decoded text and geometry agree across backends. Kernel and Radeon 780M
Vulkan admission smokes passed, with channel-affine Swish and depthwise-Swish
still selected only when measured full-boundary GPU cost is no slower than CPU.
Records: `build/bench_input_scratch_reuse_{en,diverse}_20260814.json` and
`build/smoke_input_scratch_reuse_diverse_{cpu,hybrid}_20260814.json`.

### 2026-08-14: small-to-large image memory recovery regression

The repository includes a native process-memory regression for long-lived OCR
services. `bench/generate_memory_ladder.py` produces deterministic PPM pages
that increase from `16x16` through `7680x4320`, including wide and tall
aspect-ratio cases, then process a staged `4K -> 1080p -> wide/tall -> 64x64`
recovery tail. `bench/memory_ladder_regression.py` keeps one OCR instance alive
and records Windows Private Bytes and Working Set after every resolution.

Run the full test with:

```powershell
$env:PPOCR_BENCH_THREADS = '16'
python bench/generate_memory_ladder.py --output build/memory_ladder_images
python bench/memory_ladder_regression.py `
  --cpp build/ppocr_mem_soak.exe `
  --det D:/workprj/aicoder/.tmp/ocr-models/ppocrv6_medium_det.onnx `
  --rec D:/workprj/aicoder/.tmp/ocr-models/ppocrv6_medium_rec.onnx `
  --dict build/dict_ppocrv6_latin.txt `
  --images build/memory_ladder_images --cycles 1 --backend cpu `
  --require-post-large-decline `
  --report build/memory_ladder_medium_cpu_compact_uses_20260814.json
```

On the current Windows x86 host, the test passed after the compact executor
use-count change. The `7680x4320` sample reached **150,474,752 B / 154,353,664
B** (Private Bytes / Working Set). The recovery tail fell to **147,111,936 B /
151,142,400 B** at `1920x1080`, and the final `64x64` checkpoint was
**148,066,304 B / 151,736,320 B**. Thus both counters declined below the
post-large high-water without restarting the OCR instance. The adjacent first
4K recovery measurement may briefly exceed the 8K sample because of Windows
allocator/working-set accounting; the enforced gate intentionally verifies a
decline in the complete post-large recovery tail instead of claiming a brittle
single-sample monotonic property. Full per-resolution evidence is
`build/memory_ladder_medium_cpu_compact_uses_20260814.json`.

### 2026-08-14: refreshed exact detector batching across three model sizes

The true same-shape detector NCHW batch was rerun after the memory work with
four copies of the English page. The check compares every text, quadrilateral,
bounding box, and confidence against four serial calls. Recognition retained
`rec_batch_size=1`: an investigation confirmed that forcing four recognizer
crops through the transformer can change exact CTC confidence on this host, so
it is not used for the exact-output benchmark.

With `PPOCR_BENCH_THREADS=16`, `PPOCR_DET_BATCH_SIZE=4`,
`PPOCR_REC_BATCH_SIZE=1`, and `PPOCR_BATCH_PREPROCESS_PARALLELISM=4`, the
three-run means were:

| Model | CPU serial | CPU batch | CPU speedup | Hybrid serial | Hybrid batch | Hybrid speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 133.272 ms | 76.211 ms | 1.749x | 157.468 ms | 128.652 ms | 1.224x |
| small | 282.395 ms | 177.486 ms | 1.591x | 442.434 ms | 390.907 ms | 1.132x |
| medium | 1241.956 ms | 1013.146 ms | 1.226x | 1230.259 ms | 1138.408 ms | 1.081x |

The corresponding exact-parity record is
`build/batch_detector_rec1_current_20260814.json`. Hybrid remains an
honest per-segment policy: Vulkan is selected only when the full measured
host-to-device, dispatch, and device-to-host boundary is no slower than CPU;
on this Radeon 780M run, small channel-affine Swish and depthwise-Swish
segments met that rule while the remainder correctly stayed on SIMD CPU.

### 2026-08-14: verified four-page detector NCHW batching

`RecognizeBatch()` was rechecked on four same-shape English pages for every
PP-OCRv6 tier.  The detector receives one real `[4,3,H,W]` tensor; each
result, including text, quadrilateral, bounding box, and confidence, was
checked against four serial `Recognize()` calls by `ppocr_batch_smoke`.
Recognition remains `rec_batch_size=1` in this comparison: packing unequal
recognizer widths by right-padding changes transformer attention and can alter
CTC confidence, so it is not represented as exact-output acceleration.

With 16 CPU workers, `det_batch_size=4`, four preprocessing workers, one
warm-up, and five measured CPU iterations, serial-to-batch time was:

| Model | Serial mean | Batched mean | Throughput gain |
|---|---:|---:|---:|
| tiny | 146.914 ms | 86.413 ms | 1.700x |
| small | 318.376 ms | 204.255 ms | 1.559x |
| medium | 1289.731 ms | 1011.596 ms | 1.275x |

The corresponding hybrid recheck (three iterations; the same exact-output
gate) measured **1.218x / 1.138x / 1.079x** for tiny/small/medium. Hybrid
still admits each Vulkan segment only after its complete H2D + shader + D2H
path is no slower than the matching CPU path; this is not a claim that the
incomplete full graph supports `gpu_only`. The host-specific full record is
`build/bench_batch_detector_nchw_current_20260814.json`.

Reproduce the CPU measurement with:

```powershell
$env:PPOCR_BACKEND='cpu'; $env:PPOCR_BENCH_THREADS='16'
$env:PPOCR_DET_BATCH_SIZE='4'; $env:PPOCR_REC_BATCH_SIZE='1'
$env:PPOCR_BATCH_PREPROCESS_PARALLELISM='4'; $env:PPOCR_BATCH_WARMUP='1'; $env:PPOCR_BATCH_RUNS='5'
.\build\ppocr_batch_smoke.exe DET.onnx REC.onnx dict.txt 1 en.ppm en.ppm en.ppm en.ppm
```

### 2026-08-13: Vulkan ConvTranspose + Add primitive groundwork

The Vulkan backend now has a numerically tested `ConvTranspose(2x2,
stride=2) + equal-shape Add` primitive. It keeps the projection in device
buffer D, inserts a compute write/read barrier, adds the residual in a second
dispatch, and copies only the final sum back. Its hybrid admission is strict:
it measures and validates complete `H2D + two dispatches + fence + D2H`
against the matching CPU SIMD sequence, then selects Vulkan only when no
slower.

On this Radeon 780M test shape (`N=4, Cin=32, Cout=40, 31x29`), GPU was
**2.771 ms** versus CPU **0.535 ms**, so CPU correctly remains selected. This
is backend coverage, not an end-to-end GPU speed claim. The current PP-OCRv6
FPN export folds its following `[1,C,1,1]` Add into the transpose bias, so the
new equal-shape-residual primitive is intentionally not claimed as active for
that model revision. `ppocr_vulkan_smoke` covers it; tiny/small/medium CPU
text parity passed **2/2** each on `en.ppm` and `wide_strip_3000x80.ppm`,
recorded in `build/accuracy_{tiny,small,medium}_convtranspose_add_20260813.json`.

## Build

### 2026-08-14: compact execution liveness and resolution-memory revalidation

The detector's generic executor now resets compact `uint16_t` use counters
indexed once at model load, rather than cloning a string-keyed use-count map
for every page. This removes per-inference hash-table bookkeeping while
retaining the existing last-use tensor release rules. The change applies only
to the fixed PP-OCRv6 graph topology; arbitrary graph values still retain
their original tensor ownership and output semantics.

The CPU multi-size smoke completed tiny/small/medium over five distinct
shapes (`16x16`, English, ultra-wide, ultra-tall, and dense `1920x1080`). The
record is `build/bench_diverse_compact_uses_20260814.json`; its three-run
means are intentionally retained as per-host measurement evidence rather than
portable speed claims. Kernel and Radeon 780M Vulkan admission smokes also
passed. Hybrid retains its full transfer-inclusive rule: a Vulkan segment is
used only after measured `H2D + GPU + D2H` is no slower than the same CPU
segment. GPU-only remains deliberately rejected because the complete graph is
not yet device-resident.

The 19-resolution medium CPU ladder was revalidated with one OCR instance
alive throughout. It passed `--require-post-large-decline`: the `7680x4320`
sample measured **147,972,096 B / 151,748,608 B** (Private Bytes / Working
Set), then the recovery tail reached **146,042,880 B / 148,320,256 B** at the
wide `4096x256` sample. The final `64x64` checkpoint was **147,783,680 B /
151,691,264 B**, below the largest-image private-byte value and below the
pre-return high-water. Full per-resolution evidence is
`build/memory_ladder_medium_cpu_compact_uses_final_20260814.json`.

### 2026-08-14: current ascending-resolution memory regression

The same long-lived medium CPU instance was rerun over all 19 generated
images, in ascending source sizes from `16x16` to `7680x4320`, followed by
`3840x2160 -> 1920x1080 -> 4096x256 -> 256x4096 -> 64x64`.  Each page is
scoped to `load -> Recognize -> destroy -> sample`, so the measurements expose
both inference activation reuse and whether a decoded large source image stays
resident after its OCR call.

The strict recovery gate passed.  At 8K, Private Bytes / Working Set were
**148,131,840 B / 152,174,592 B**.  The immediately following 4K page was
**147,681,280 B / 151,670,784 B**, a drop of **450,560 B / 503,808 B**.  The
final `64x64` checkpoint was **147,587,072 B / 151,429,120 B**, below both the
8K sample and the pre-return high-water (**150,106,112 B / 154,062,848 B**),
without recreating the OCR instance.  Per-resolution samples and the enforced
pass/fail decision are recorded in
`build/memory_ladder_medium_cpu_current_resolution_recovery_20260814.json`.


### 2026-08-14: refreshed three-tier CPU / Hybrid comparison

`bench/compare.ps1` now accepts `-Backend cpu|hybrid|gpu` and passes the
choice through to the native benchmark. This makes the CPU and Hybrid runs
explicit in the same ONNX Runtime comparison harness instead of inheriting an
ambient process environment.

On the local AVX-512 host, `en.ppm`, 2 warm-ups and 6 measured runs, the
native CPU implementation measured **32.315 ms** (tiny) and **67.419 ms**
(small), versus ONNX Runtime CPUExecutionProvider **113.453 ms** and
**149.804 ms**: **71.5%** and **55.0%** lower mean latency respectively. The
same harness measured medium Hybrid at **239.765 ms** versus ONNX Runtime
**333.395 ms** (**28.1%** lower). These are machine/run-specific samples;
reproduce before treating them as deployment guarantees.

The medium high-density `dense_blobs` stress bound was refreshed to
`[100, 3600]`. The current native medium detector produced 2,903 valid
components and the full 11-image varied-size stress suite passed.

### 2026-08-14: terminal CTC projection is now the default

The recognizer's final vocabulary projection is now executed through the
bounded `GemmCtcTop1` route by default. It computes each small time-step tile,
performs the CTC argmax/probability reduction, and releases that tile rather
than materializing the full `[N,T,V]` logit tensor. Set
`PPOCR_DISABLE_FUSED_TERMINAL_CTC=1` only for an A/B comparison with the older
materialized-logit route.

On the dense `ui_dense_1920x1080` corpus the default route reduced measured
CPU latency in this local run from **279.538 to 263.995 ms** for tiny
(123 crops), **1077.122 to 965.587 ms** for small (119 crops), and
**6285.383 to 5621.346 ms** for medium (122 crops). These samples are
machine-specific; text parity against ONNX Runtime passed 4/4 varied inputs
for each model after enabling the default.

### 2026-08-14: refreshed 16x16-to-8K memory recovery measurement

The deterministic 19-image ladder was rerun with one live medium CPU OCR
instance. It processes ascending source resolutions from `16x16` through
`7680x4320`, then a `3840x2160 -> 1920x1080 -> 4096x256 -> 256x4096 ->
64x64` recovery tail. Every input is scoped to `load -> Recognize -> destroy
-> sample`, so this measures both activation reuse and release of a large
decoded RGB source image.

The strict post-large recovery gate passed. The 5K/8K high-water was
**150,675,456 B / 154,615,808 B** (Private Bytes / Working Set); the first
4K recovery sample was **150,036,480 B / 154,087,424 B**, and the final
`64x64` sample was **148,877,312 B / 151,830,528 B**. The final small-image
footprint was 3.63% above the warmed private-byte baseline and below both
pre-return peaks, confirming bounded memory recovery without recreating the
OCR instance. Full per-resolution evidence is
`build/memory_ladder_medium_cpu_post_large_20260814.json`. Windows allocator
accounting may fluctuate between adjacent samples; the enforced condition is
that both counters return below the 5K/8K high-water somewhere in the
post-large recovery tail.
### 2026-08-14: CTC tile-policy validation and final memory recheck

The four-row bounded terminal-CTC projection remains the default. An
experimental automatic eight-row choice for the wider recognizers was tested
on the high-density `1920x1080` sample, but was slower in the paired local
runs: small measured **914.427 ms** versus **897.369 ms** with four rows, and
medium measured **4829.971 ms** versus **4529.301 ms**. The larger tile
remains available only through `PPOCR_FUSED_TERMINAL_CTC_TILE_ROWS` for
platform-specific experiments; it is not used as the portable default.

After this decision, the complete 19-resolution medium CPU memory check
passed again. Private Bytes / Working Set fell from **147,623,936 B /
151,322,624 B** at the 8K sample to **146,046,976 B / 149,807,104 B** at the
first 4K recovery; final `64x64` was **143,917,056 B / 147,906,560 B**
(**0.04%** above the warmed private-byte baseline), below the full pre-return
high-water. The durable per-resolution record is
`build/memory_ladder_medium_cpu_tile4_20260814.json`.
### 2026-08-14: recognizer hotspot profiling and Vulkan Hybrid revalidation

`PPOCR_PROFILE=1` now reports the terminal-CTC executor even when its fused
`GemmCtcTop1` path returns early. On the medium ultra-wide crop, that direct
measurement identifies the terminal vocabulary projection as the largest
recognizer operation (**40.456 ms** in this local profile), followed by the
existing fused 1x1/residual convolutions. This makes future optimization work
measurable instead of inferring recognizer cost from detector-only profiles.

The Vulkan Hybrid policy was revalidated on AMD Radeon 780M. It measures each
implemented segment as a complete synchronous `H2D + dispatch + D2H` path and
selects GPU on `gpu_mean <= cpu_mean`; smoke selected GPU for channel-affine
Swish (**0.193 vs 0.404 ms**) and depthwise-Swish (**0.341 vs 0.346 ms**),
while transfer-bound primitives correctly remained on CPU. Tiny, small, and
medium Hybrid text parity all passed on normal and ultra-wide inputs (6/6).
GPU-only remains deliberately unavailable until the complete detector,
recognizer, and post-processing graph is device-resident; it does not silently
fall back to CPU.

The AVX-512 four-row GEMM microkernel was also retested against the stable
two-row default. It remains an explicit
`PPOCR_ENABLE_AVX512_GEMM4ROW=1` deployment A/B option only: its effect varied
between otherwise similar paired runs, so it is not enabled by default.

### 2026-08-14: medium terminal-CTC projection parallelism

The terminal CTC projection now uses a bounded **32-row** vocabulary tile for
the medium recognizer (`depth >= 384`, `vocab >= 1024`); tiny and small retain
the prior four-row tile. This gives a single ultra-wide medium crop enough
independent GEMM rows to use the persistent SIMD pool, while keeping CTC
blank/repeat state serial in time order and the workspace bounded. The
`PPOCR_FUSED_TERMINAL_CTC_TILE_ROWS=1|2|4|8|16|32` override remains available
for deployment-side A/B.

On this AVX-512 Windows host (16 workers), paired ultra-wide `3000x80`
measurements improved medium from **189.408 ms** (four rows) to **150.969 ms**
(32 rows), a **20.3%** reduction; small improved from **61.882 to 56.759 ms**
in that isolated wide test, but dense-page runs were less stable, so its
conservative default remains four rows. Medium dense `1920x1080` also improved
from **5237.145 to 4671.601 ms** in the paired run. These figures are local
host measurements, not portable latency guarantees.

The associated 19-resolution medium CPU memory gate passed: 8K was
**150,437,888 B / 154,501,120 B** (Private Bytes / Working Set), immediate
4K recovery was **148,398,080 B / 152,215,552 B**, and final `64x64` was
**144,076,800 B / 148,070,400 B**, below baseline and all large-page
high-water marks. Per-resolution evidence:
`build/memory_ladder_medium_cpu_ctc32_20260814.json`.

The same gate was rerun after the final CTC policy validation. It again
passed the enforced `--require-post-large-decline` condition: the 5K/8K
high-water was **149,454,848 B / 153,169,920 B**, the first post-8K 4K page
fell to **146,534,400 B / 150,233,088 B**, and the final `64x64` checkpoint
was **147,755,008 B / 151,703,552 B**. This is **0.37%** above the warmed
**147,210,240 B** private-byte baseline and below both pre-return peaks, so
large decoded images and inference workspaces do not remain pinned after the
large-image sequence. The ladder deliberately covers 19 inputs from `16x16`
through `7680x4320`, including wide/tall aspect-ratio branches and a staged
`4K -> 1080p -> wide -> tall -> 64x64` recovery tail. Full JSON evidence:
`build/memory_ladder_medium_cpu_ctc32_recheck_20260814.json`.

### 2026-08-14: CTC workspace reuse experiment rejected

A CTC vocabulary-tile reuse experiment was measured on the 123/119/122-crop
dense `1920x1080` page. Sharing one preallocated tile per in-flight sequence
increased the medium batch workspace enough to cause a memory-recovery gate
failure on one fresh run, while its latency did not improve reliably. The
production implementation therefore retains a short-lived, per-worker bounded
tile: it has a strict `32 * vocab`-float cap for medium and `4 * vocab` for
tiny/small, releases promptly after each recognition task, and avoids unsafe
MinGW thread-local destructors. This choice is deliberately evidence-led: the
restored implementation passed the 19-resolution CPU memory gate again, with
the first recovery-tail point below the 8K private/working-set high-water at
`1920x1080` (**146,456,576 B / 149,090,304 B**) and final `64x64` still below
the pre-return peak. Evidence:
`build/memory_ladder_medium_cpu_ctc_workspace_reverted_20260814.json`.

### 2026-08-14: hardware-aware recognition worker bound

The 12-way CPU recognition default is capped at
`std::thread::hardware_concurrency()`. This retains the measured 12-way
policy on the local 16-logical-core AVX-512 machine while preventing a small
x86 or ARM target from constructing more independent crop graphs than its
available logical CPUs. Hybrid deliberately keeps its separate
`hybrid_graph_parallelism` policy because its Vulkan queue is serialized. The
public `PPOCR_REC_PARALLELISM` override remains available for host-specific
tuning.

### 2026-08-14: dense-page recognition scheduler tuning

The CPU default `rec_parallelism` is now **12** (override with
`PPOCR_REC_PARALLELISM`). A 16-worker AVX-512 sweep on the 123/119/122-crop
dense `1920x1080` page measured 8-way versus 12-way versus 16-way outer
recognition scheduling. Twelve was the stable cross-model choice: tiny
improved from **282.675 to 241.928 ms** (14.4%), small from **949.373 to
**885.617 ms** (6.7%), and medium from **3968.286 to 3905.875 ms** (1.6%).
At 16 ways medium regressed to **4375.612 ms**, so the default deliberately
leaves headroom for its larger SIMD kernels and avoids multiplying dense-page
activation lifetime further.

A fresh default run measured **236.679 / 854.346 / 4136.895 ms** for
tiny/small/medium respectively (3/3, 3/3, and 2/2 measured samples). Normal
and ultra-wide decoded-text parity against ORT remained 6/6. The 19-resolution
medium memory gate also passed: after the 8K page, Private Bytes / Working
Set fell from **149,106,688 / 153,096,192 B** to **147,845,120 /
151,638,016 B** at the first 4K recovery page; the final `64x64` checkpoint
remained below the large-page high-water. Evidence:
`build/memory_ladder_medium_cpu_recparallel12_20260814.json`.

### 2026-08-14: multi-image native/ORT gate repair and revalidation

`bench/compare_models.ps1` now reliably expands a comma-separated or native
PowerShell `string[]` image list, forwards the selected CPU/Hybrid backend,
and distinguishes a latency-weighted corpus gate from an explicit per-case
floor. This prevents a one-image invocation from being presented as a
multi-resolution result and makes each model/image pair independently auditable.

A fresh CPU validation used normal English and ultra-wide `3000x80` text,
16 workers, one warm-up, two timed runs, and required every one of the six
model/image pairs to beat ONNX Runtime CPU. All passed. The latency-weighted
native reductions were **68.1%** (tiny), **56.2%** (small), and **22.9%**
(medium); no model had a losing case in this run. Full machine-specific
evidence: `build/compare_models_cpu_normal_wide_gate_20260814.json`.

The same wrapper also supports `-Backend hybrid`; on this Radeon 780M UMA
host, it preserves the transfer-inclusive Vulkan admission policy rather than
claiming universal end-to-end GPU gains.

### 2026-08-14: refreshed three-tier ONNX Runtime comparison

The current 16-worker CPU comparison (2 warm-ups, 6 measured `en.ppm` runs)
continues to show substantial native gains over ONNX Runtime CPUExecutionProvider:
**33.084 vs 118.015 ms** for tiny (**72.0% lower**), **72.522 vs 195.894 ms**
for small (**63.0% lower**), and **222.385 vs 402.871 ms** for medium
(**44.8% lower**). These are paired local-host measurements and include the
complete native OCR pipeline, not an isolated kernel claim.

An additional selected-class CTC-softmax parallelization experiment was
rejected: it was neutral on a single wide crop and slower on a dense medium
page, so the shipped path retains the lower-variance temporal CTC scan.
Hybrid text parity was rerun for tiny/small/medium on normal and ultra-wide
inputs (6/6); the GPU decision rule remains unchanged: choose Vulkan only
when the complete transfer-inclusive segment is no slower than CPU.

### 2026-08-14: Vulkan device-resident graph foundation

The Vulkan backend now has a `VulkanTensorArena` separate from Hybrid's five
short-lived scratch bindings. It allocates FP32 storage slots by tensor size,
reuses a released slot only after the caller's final-use decision, and reports
live versus retained capacity. Graph boundaries can upload inputs/constants
and download requested outputs, while intermediate elementwise values remain
in Vulkan buffers. The first device-resident operators are exact binary
in-place updates and the NCHW 1x1 Conv family used throughout the recognizer;
both insert compute write-to-read barriers before their output can feed the
next graph node.

The Radeon 780M Vulkan smoke now verifies slot allocation/reuse, upload,
device binary and dependent binary-chain execution, plus device-resident
pointwise convolution against CPU FP32 reference values. The regular SIMD and
all existing Hybrid primitives remain covered. This is infrastructure for the
complete GPU-only graph, not a claim that `gpu_only` is available yet: pool,
LayerNorm, attention Softmax/QKV, transposes, concatenation, detector DB
post-processing, and remaining graph dispatches still need device paths.

### 2026-08-14: GPU-only graph operator expansion

The device arena now additionally supports a real destination-writing NCHW
suffix broadcast (`[N,C,H,W]` with compact `[C,1,1]` coefficients) and exact
Erf-form GELU on the shader. Both are numerically checked by
`ppocr_vulkan_smoke` on the AMD Radeon 780M; neither reads an intermediate
activation back through the CPU. The broadcast form is required for SE gates
and compact graph scale paths, while exact GELU covers a fused-model operator
present in every supported detector/recognizer pair.

This does **not** enable `--backend gpu`: the complete graph still lacks
device implementations for graph concatenation/shape-slice metadata handling,
unfused resize forms, SE composition, and the full OCR image/DB/CTC pipeline.
Construction continues to fail explicitly rather than silently executing an
activation on CPU. No end-to-end GPU-only benchmark is recorded until that
contract can be exercised and compared fairly.

The same arena now includes device-resident two-to-four input `Concat` and
integer-factor asymmetric/floor nearest `Resize` (including unequal height /
width factors). These are validated with multi-outer-block and `N=2` smoke
cases. They remove two more detector-FPN activation boundaries; they do not
change the strict GPU-only availability gate because a complete model graph
and OCR front/back-end still require verified device dispatches.

Medium-model SE is now represented as one device-resident arena routine:
spatial mean, both 1x1 projections, ReLU, HardSigmoid, and final channel
multiply all remain in Vulkan buffers. The Radeon smoke checks every output
against an independent FP32 reference for `N=2,C=4,reduced-C=3,plane=13`.

The ONNX executor now also has a strict Vulkan graph-execution scaffold with
device value metadata, graph-use-count slot reclamation, device copy for
branch fan-out, metadata-only reshape/squeeze/unsqueeze, pointwise unary and
binary aliases, Concat, and fused SE routing. It returns failure immediately
on an uncovered graph node; the public GPU-only gate remains closed until the
entire detector, recognizer, and OCR pipeline can use that path.

The strict graph executor now maps normal/fused NCHW Conv and Max/AveragePool
to their device-arena implementations. When an activation fan-out prevents an
in-place update, it first uses the new GPU copy primitive; this preserves
graph arithmetic rather than overwriting a still-live branch. Unsupported
geometry or nodes still terminate GPU-only execution instead of invoking CPU.

### 2026-08-15: strict Vulkan graph coverage extended

The strict device graph now also covers detector depthwise/pointwise blocks,
residual 1x1 projections, compact SE-style broadcasts, spatial means,
integer nearest resize and resize-add, 2x2 stride-two transpose convolution,
SAME padding for Conv/Pool, inference BatchNorm channel-affine, rank 2--4
transpose, and final matrix projection/Softmax. `FusedConvHardSigmoid` was
corrected to run its exported alpha/beta activation as a separate Vulkan unary
after convolution; it no longer risks returning an ungated Conv result.

`ppocr_gpu_graph_smoke` was added as a direct CPU-versus-strict-device graph
tool. It deliberately does not mark public `gpu_only` OCR as available: the
full image preprocessing, DB detector postprocessing, crop generation and
CTC decoding must still be moved onto Vulkan before that user-facing contract
can be truthfully enabled. On 2026-08-15 the Radeon 780M low-level Vulkan and
SIMD smoke suites passed; full-graph parity is still gated while the remaining
recognizer attention nodes and final numerical-error envelope are completed.

### 2026-08-15: GPU-only recognizer scheduling progress

The strict Vulkan executor now runs the PP-OCRv6 small/medium recognizer's
dynamic shape-metadata chains (`Shape`/`Slice`/`Squeeze`/`Unsqueeze`/`Concat`),
scalar attention scale, fused QKV split, rank-3/rank-4 attention `MatMul`,
Softmax, and residual transformer blocks without downloading an activation.
Only integer shape descriptors are interpreted on the host; feature maps stay
in arena buffers. The generic GEMM path is the current correctness baseline;
the experimental tiled GEMM requires `PPOCR_ENABLE_VULKAN_GEMM_TILE=1`.

Verification on the AMD Radeon 780M: `ppocr_vulkan_smoke` and
`ppocr_kernel_smoke` pass, and the normal tiny Hybrid OCR benchmark remained
functional at **10.775 ms** mean over two measured runs on `line_en.ppm`.
The ordinary-Conv shared-memory route is now gated by the same large-output
threshold in host dispatch and shader dispatch. Previously a compact SE gate
could be launched through the generic host geometry but interpreted as a
channel-tiled shader geometry, leaving stale arena values. With that mismatch
removed, strict CPU-versus-Vulkan forward parity now passes for every shipped
network on the Radeon 780M: tiny/small/medium detector at `1x3x960x960`, and
tiny/small/medium recognizer at `1x3x48x320`. The largest observed final
absolute errors are `2.42644e-06`, `6.57933e-06`, and `7.00675e-09` for the
detectors, and `5.88894e-05`, `2.09391e-04`, and `8.39829e-05` for the
recognizers, respectively.

`ppocr_gpu_graph_smoke` now accepts an optional `RUNS` argument and prints
CPU/GPU graph means, including the current GPU graph boundary. The initial
two-run reference on this UMA Radeon machine measured tiny detector
`124.284 / 610.187 ms`, tiny recognizer `8.190 / 173.200 ms`, small detector
`219.862 / 1613.830 ms`, small recognizer `20.697 / 370.361 ms`, medium
detector `1033.580 / 8530.580 ms`, and medium recognizer `33.085 / 577.311
ms` (CPU / strict Vulkan). These are correctness-baseline measurements, not
a GPU speed claim: each current graph operator submits and waits separately,
so command-batch compilation and persistent model constants are the next
Vulkan performance work.

### 2026-08-15: strict graph single-submit command recording

The strict neural Vulkan executor now records a complete forward graph into
one command buffer and waits only at its output boundary. Every dispatch gets
an immutable descriptor set from a dedicated resettable graph pool, preventing
later descriptor writes from changing earlier recorded bindings. Hybrid keeps
its original synchronous descriptor pool and full-boundary admission policy.

The local Radeon 780M verification passed the Vulkan primitive suite, SIMD
kernel suite, and all six strict graph parity cases at the published shapes.
The current two-run CPU / Vulkan means are tiny detector **223.271 / 1042.440
ms**, tiny recognizer **29.517 / 319.981 ms**, small detector **195.110 /
1249.390 ms**, small recognizer **18.378 / 291.485 ms**, medium detector
**812.492 / 10669.200 ms**, and medium recognizer **39.293 / 564.885 ms**.
Final tensor parity was exact in these runs (max absolute error `0`). The UMA
driver still makes this graph implementation slower than the AVX-512 CPU
baseline; eliminating submission fences alone is not enough to claim a
speedup, so the public GPU-only OCR contract remains closed.

### 2026-08-15: persistent GPU model weights and lower Conv tile threshold

Strict `OnnxLite` GPU models now retain their immutable initializer slots in a
model-owned Vulkan arena. Activations are released after every successful or
failed run, and the arena is guarded for the complete graph recording; this
removes repeated model-weight allocation/upload and prevents per-inference
arena growth. The ordinary-Conv shared-filter tile threshold was lowered from
16,384 to 4,096 output pixels in both host and shader dispatch logic, while
retaining the compact SE fallback path.

With a warmed model, Radeon 780M three-run strict graph means were **tiny
detector 111.399 / 264.214 ms**, **small detector 229.969 / 858.226 ms**,
**medium detector 820.237 / 8059.020 ms** (medium: two runs), **tiny
recognizer 7.915 / 86.976 ms**, **small recognizer 18.101 / 122.982 ms**, and
**medium recognizer 34.873 / 354.897 ms** (CPU / Vulkan). Exact parity held
for all six models. The tiny detector resolution ladder also passed at
`320²`, `640²`, and `960²` with zero final error; warmed Vulkan means were
**40.471**, **109.382**, and **264.214 ms**, respectively. A real `N=4`
tiny-recognizer batch at `4x3x48x320` also remained exactly equal to CPU.

These results are a substantial reduction of strict-device scheduling and
weight-transfer overhead, but this Radeon 780M UMA configuration is still not
faster than its AVX-512 CPU path for complete PP-OCRv6 graphs. Hybrid retains
its measured per-segment policy: when Vulkan is equal or faster for the full
boundary it is selected, otherwise CPU is retained to avoid consuming GPU
time without a latency benefit.

### 2026-08-15: expanded Conv tile range

The ordinary-Conv shared-weight tile is now eligible from a `256`-pixel
output plane (host and SPIR-V use the same gate), extending reuse to smaller
detector feature maps without changing compact SE convolutions. The regenerated
shader and strict graph test passed with final error `0` on tiny detector
`320x320` and `960x960`; warmed Vulkan means were **37.13 ms** and
**236.52 ms**, respectively. Lowering the threshold again to `64` regressed
both shapes, so `256` is retained.

### 2026-08-15: GPU-only front/back-end groundwork

The Vulkan arena now has a live-slot graph handoff API, fused BatchNorm+Swish
dispatch, RGB bilinear-resize+NCHW-normalization shader, device threshold-mask
shader, and device CTC Top-1 reduction. `ppocr_gpu_frontend_smoke` verifies
the RGB path on Radeon 780M; its maximum detector-normalized difference from
the scalar front-end was **0.0175072**, below one source uint8 normalization
step. The strict tiny detector and recognizer graph regressions remained exact
at their final tensors.

This is deliberately not advertised as a completed public GPU-only OCR mode:
DB box scores/unclip parity and whole-image decoded-text parity still require
an end-to-end validation suite. Accordingly `Backend::gpu_only` continues to
fail explicitly rather than returning potentially different OCR results. The
experimental device orchestration never falls back to the CPU NN executor,
but it is not enabled through the public API until that parity evidence exists.

### 2026-08-15: strict graph memory ladder and default recognizer GEMM tile

`ppocr_gpu_graph_mem_soak` is a new model-level Vulkan memory regression tool.
It warms one model, drives it through an ascending resolution/batch ladder,
then repeats a small input and reports both allocated arena capacity and live
bytes. Capacity is deliberately reusable high-water storage; the required
invariant is that live bytes return to the model's persistent-weights baseline
after every run.

On the Radeon 780M, tiny detector `320x320 -> 640x640 -> 960x960 -> 320x320`
grew capacity to **214,007,808 bytes** while live bytes returned after every
sample to **1,713,620 bytes**. Tiny recognizer
`1x3x48x80 -> 1x3x48x320 -> 4x3x48x320 -> 8x3x48x640 -> 1x3x48x80` reached
**67,633,152 bytes** and recovered each time to **4,417,112 bytes**. The
same invariant passed on small detector (**303,267,840 / 9,813,412 bytes**)
and medium recognizer (**191,889,408 / 76,456,032 bytes** capacity/live
baseline). This explicitly covers both varied detector images and recognition
batch/width growth without retaining dynamic activations after a large run.

The verified non-Swish strict-graph GEMM tile is now default-enabled for
`rows >= 8, depth >= 16, columns >= 128`; set
`PPOCR_DISABLE_VULKAN_GEMM_TILE=1` for an A/B fallback. On the same host,
final tensor parity remained exactly zero for all three recognizers. The
current short strict-recognizer means at `1x3x48x320` were tiny
**9.869 / 37.195 ms**, small **23.498 / 107.000 ms**, and medium
**45.271 / 389.620 ms** (CPU / Vulkan). This is an improvement for the
device graph but not a claim that this AVX-512 + Radeon 780M UMA system beats
CPU or ONNX Runtime end-to-end; Hybrid still uses its full-boundary
no-slower admission rule.

### 2026-08-15: parallel Softmax writeback

The strict Vulkan Softmax now retains its sequential max/sum reduction for
numerical stability, then spreads the independent exponential/writeback stage
across the 256-lane workgroup. This removes a serial device loop in
recognizer attention and terminal class Softmax without adding an activation
allocation. Final tensor parity remained **0** for tiny, small, and medium
recognizers at `1x3x48x320`, and at a real `4x3x48x320` batch. The short
post-change means were tiny **37.470 ms**, small **94.376 ms**, and medium
**382.981 ms** for the single-crop strict Vulkan graph; the small recognizer
improved from the preceding **107.000 ms** sample. Device output is still
slower than the local AVX-512 CPU baseline, so this is a verified incremental
improvement rather than an ONNX Runtime superiority claim.

The public `gpu_only` OCR switch remains deliberately disabled. Image resize
and normalization upload, DB connected-component/unclip postprocessing, crop
generation, CTC top-1/decode, end-to-end output parity, and memory recovery
must be completed before the public contract can truthfully promise complete
GPU-only OCR.

### 2026-08-15: device-resident OCR path validation gate

The experimental OCR route now feeds strict Vulkan detector/recognizer graphs
from the fused Vulkan RGB resize, BGR conversion and normalization dispatch;
it no longer builds detector or recognizer NCHW activations with CPU floating
point code. Detector probabilities are downloaded for the existing DB
connected-component/score/unclip control path, and only compact CTC Top-1
metadata is downloaded for string assembly. `ppocr_gpu_ocr_smoke` compares
the public OCR result (text, quad, bounding rectangle and confidence) with
CPU-only OCR.

On the Radeon 780M, tiny and small models passed `en.ppm`, `line_en.ppm` and
`mixed.ppm` (6 decoded regions each), with maximum confidence differences
of **0.0000061** and **0.0000019**, respectively. Tiny also passed `en.ppm`
alone at **0.0000026**. These runs establish that the NN graph has no silent
CPU operator fallback, but do not establish complete release coverage:
the dense 1920x1080 page diverged in detector/recognizer results (tiny result
72; small result 16), and medium detector strict submission at its real
`1x3x160x704` shape produced an all-zero probability map (while its 128x704
pipeline smoke passed). Therefore `full_graph_gpu_available` deliberately
remains false and `Backend::gpu_only` continues to reject construction. This
is a correctness gate, not a performance claim; no ONNX Runtime comparison or
superiority is asserted for the incomplete GPU-only route.

### 2026-08-15: GPU-only completion status

The implementation now has an all-device neural graph path, fused RGB-to-NCHW
front end, device CTC Top-1 and no CPU floating-point inference fallback.
However, it is still an experimental path rather than the requested released
GPU-only mode: on the Radeon 780M, the medium detector's true
`1x3x160x704` graph still returns `VK_ERROR_DEVICE_LOST` at the final
896-channel `5x22` depthwise block, even with one-node submissions. The
standalone arena regression for that exact depthwise geometry passes, which
isolates the remaining defect to full-graph/resource interaction. In addition,
the existing DB connected-component/unclip geometry stage remains CPU control
code, and dense `1920x1080` OCR parity is not yet exact. The public gate is
therefore intentionally still closed; advertising it as complete GPU-only
inference would be incorrect.

### 2026-08-15: GPU-only revalidation (current workspace)

The strict Vulkan implementation was rebuilt after repairing the source file
and revalidated on **AMD Radeon 780M Graphics** with
`PPOCR_GPU_ONLY_SEGMENT_NODES=1`.  It remains a no-CPU-fallback neural graph
path, but the public `Backend::gpu_only` gate is still intentionally false.
The following current CPU-versus-device graph checks passed:

| Model | Shape | Max absolute output error | CPU mean | Vulkan mean |
|---|---:|---:|---:|---:|
| tiny detector | `1x3x160x704` | `4.83182e-07` | 28.844 ms | 224.343 ms |
| small detector | `1x3x160x704` | `1.91038e-07` | 41.481 ms | 301.020 ms |
| medium recognizer | `1x3x48x320` | `8.39829e-05` | 60.583 ms | 552.145 ms |

The Vulkan primitive suite also passes, including the standalone
`C=896,H=5,W=22` depthwise convolution that resembles the medium detector
terminal block.  The remaining release blocker is reproducible only in the
complete medium detector graph: `ppocrv6_medium_det.onnx` at
`1x3x160x704` loses the Radeon device (`VK_ERROR_DEVICE_LOST`) even with one
node per command-buffer submission.  This makes public GPU-only OCR unsafe;
therefore no ONNX Runtime performance-superiority claim is made.

`ppocr_gpu_graph_mem_soak` now provides a device-arena recovery check.  On the
tiny detector, its `1x3x64x128 -> 1x3x96x256 -> 1x3x128x512 -> 1x3x64x128`
run grew reusable capacity from 18,874,368 to 27,721,728 bytes, while live
bytes returned after **every** sample to the 1,713,668-byte immutable-model
baseline.  Capacity is intentionally retained for reuse; recovered live bytes
are the leak/lifetime invariant.
## Model files

Get `ocr-models.zip` from the MaClaw `Model_Release` and unpack it.  Pass a
matching detector, recognizer, and dictionary.  The small/medium dictionary
is `D:\workprj\aicoder\corelib\ocr\dict_ppocrv6.txt`; the model archive
contains `dict_ppocrv6_tiny.txt` for the tiny recognizer.

## Demo

The included demo uses binary P6 PPM to avoid coupling a specific image
decoder to the inference library:

```powershell
.\build\ppocr_demo.exe ppocrv6_tiny_det.onnx ppocrv6_tiny_rec.onnx dict_ppocrv6_tiny.txt image.ppm
```

For PNG/JPEG/BMP, decode with the host application's existing image library
and construct `ppocr::Image { width, height, packed_rgb }` before calling
`ppocr::OCR::Recognize`.

## Scope

The C++ pipeline contains PP-OCRv6 BGR preprocessing, dynamic detection and
recognition sizing, CTC greedy decoding, and DB-style threshold/score/unclip
text region extraction.  Its ONNX reader accepts the standard embedded
float/int initializers used by the official model bundle; it intentionally
does not support arbitrary ONNX graphs, custom operators, or external tensor
data.

### 2026-08-15: strict GPU-only execution revalidation

The strict Vulkan graph executor now uses a device-resident submission boundary
per graph node by default. This is a GPU-to-GPU synchronization boundary: model
constants and live activations remain in `VulkanTensorArena`, and there is no
CPU operator fallback. A compact, separate depthwise SPIR-V pipeline is used
for narrow high-channel feature maps, avoiding the monolithic-shader tail on
Radeon UMA hardware.

On AMD Radeon 780M, CPU versus strict GPU graph checks passed for:

| Model | Shape | Max absolute output error | CPU mean | Vulkan mean |
|---|---:|---:|---:|---:|
| tiny detector | `1x3x160x704` | `4.83182e-07` | 22.184 ms | 113.042 ms |
| small detector | `1x3x160x704` | `1.91038e-07` | 42.603 ms | 193.311 ms |
| medium detector | `1x3x160x704` | `6.56222e-04` | 128.313 ms | 757.807 ms |
| medium recognizer | `1x3x48x320` | `8.39829e-05` | 39.960 ms | 472.034 ms |

The medium detector memory ladder (`64x128 → 96x256 → 128x512 → 160x704 →
64x128`) also passed: its grow-only arena reached `291,667,968` bytes, while
live allocation returned after every run to the immutable-model baseline of
`61,946,548` bytes.

Public `Backend::gpu_only` remains deliberately unavailable. Tiny and small
completed end-to-end CPU/GPU OCR parity on `en.ppm` (maximum confidence errors
`0.0000029` and `0.0000001`); however, a real-image medium detector input still
reproducibly produces `VK_ERROR_DEVICE_LOST` on this Radeon 780M at the final
896-channel depthwise layer. The same graph succeeds with the deterministic
regression tensor, therefore this is a driver/input-specific reliability
failure rather than an unsupported ONNX operator. Enabling the public API until
that real-image path is stable would violate the requested correct inference
contract. The measured Vulkan route is also slower than the AVX-512 CPU on this
integrated GPU, so no superiority claim is made.

### 2026-08-15: GPU-only arena lifetime and current recheck

Strict Vulkan graph segments now close only after their final input consumers
have been retired.  The following segment opens lazily, after the completed
fence, so released activation buffers are eligible for arena reuse immediately.
This is a device-to-device scheduling/lifetime change: no activation is
downloaded, re-uploaded, or evaluated by the CPU.

Current Radeon 780M checks with `PPOCR_GPU_ONLY_SEGMENT_NODES=1`:

| Model | Shape | Max absolute output error | CPU mean | Vulkan mean |
|---|---:|---:|---:|---:|
| tiny detector | `1x3x160x704` | `4.83182e-07` | 19.293 ms | 100.492 ms |
| small detector | `1x3x160x704` | `1.91038e-07` | 29.877 ms | 187.476 ms |
| medium detector | `1x3x128x704` | `7.48696e-09` | 85.258 ms | 716.522 ms |
| medium recognizer | `1x3x48x320` | `8.39829e-05` | 40.103 ms | 481.122 ms |

The tiny-detector device memory ladder
`64x128 -> 96x256 -> 128x512 -> 64x128` recovered to its
`1,713,668`-byte immutable-constant baseline after every sample.  Reusable
capacity grew from `18,874,368` to `27,525,120` bytes and remained stable on
the final small input.

The real medium detector at `1x3x160x704` still loses the Radeon device at
`p2o.pd_op.depthwise_conv2d.12.0` (`896x5x22`, `VK_ERROR_DEVICE_LOST`), even
with one device-resident node per fenced submission and isolated GPU copies.
Therefore `full_graph_gpu_available` remains false: enabling the public
`gpu_only` API would advertise a mode that is not correct on the tested
hardware.  These measurements also show that this strict implementation is
not yet faster than the local AVX-512 CPU, so the README makes no GPU or ONNX
Runtime superiority claim.

`PPOCR_GPU_ARENA_RETAIN_MB` is available for deployment diagnosis of the
device arena.  Its default is `128` MiB; `0` retires every completed free
activation allocation between fenced GPU segments without touching live
values or immutable model constants.  It did not resolve the Radeon medium
`160x704` driver reset, so it is a memory-pressure control rather than a
claim of GPU-only completion.

### 2026-08-15: GPU-only reliability status (latest)

The strict Vulkan graph remains fully device-resident between its input and
output boundaries: all PP-OCRv6 neural operators execute in Vulkan buffers,
and failure is reported rather than silently executing a CPU operator. The
experimental generated `vulkan_depthwise_tail.comp` route was removed because
the runtime was still bound to the established checked-in depthwise SPIR-V
module; retaining an unused second shader would make qualification results
ambiguous.

Current Radeon 780M verification with `PPOCR_GPU_ONLY_SEGMENT_NODES=1`:

| Model | Shape | Max absolute output error | CPU mean | Vulkan mean |
|---|---:|---:|---:|---:|
| tiny detector | `1x3x160x704` | `4.83182e-07` | 21.400 ms | 103.202 ms |
| small detector | `1x3x160x704` | `1.91038e-07` | 35.271 ms | 182.782 ms |
| medium detector | `1x3x128x704` | `7.32689e-09` | 103.195 ms | 779.817 ms |
| medium recognizer | `1x3x48x320` | `8.43406e-05` | 42.774 ms | 492.075 ms |

The complete medium detector at its production `1x3x160x704` shape still
reliably reports `VK_ERROR_DEVICE_LOST` at the final `C=896, H=5, W=22`
depthwise block. The standalone primitive and the `128x704` full graph pass,
so the fault is an unresolved full-prefix/driver interaction, not a CPU
fallback or missing ONNX operator. `Backend::gpu_only` therefore remains
intentionally unavailable. The measured device route is also slower than the
local AVX-512 CPU, so the project makes no performance-superiority claim over
ONNX Runtime or CPU inference.

### 2026-08-15: GPU-only qualification recheck

The strict neural-graph implementation remains pure Vulkan: activations and
model constants stay in `VulkanTensorArena`, and a failed device graph throws
instead of falling back to the CPU executor.  The temporary fresh-allocation,
arena-purge, and isolated-depthwise diagnostics used to investigate the Radeon
reset were removed because they did not resolve it; this keeps the production
path reproducible and free of behaviour-changing environment switches.

The current clean build (`cmake -S . -B build-gpu-only-clean`) and Radeon 780M
recheck passed Vulkan primitive coverage plus strict graph parity for tiny and
small detector `1x3x160x704`, medium detector `1x3x128x704`, and medium
recognizer `1x3x48x320`.  The GPU RGB front-end -> detector hand-off passed on
the real `en.ppm` page for tiny and small.  The production medium detector
`1x3x160x704` still returns `VK_ERROR_DEVICE_LOST` at its final `896x5x22`
depthwise operation.  Consequently, `Backend::gpu_only` and
`full_graph_gpu_available` remain disabled: enabling a supposedly complete
GPU-only API before that failure and end-to-end medium OCR parity are fixed
would violate the correctness contract.  No GPU/ONNX Runtime speed claim is
made for this UMA adapter.

### 2026-08-15: Vulkan full-OCR requalification and MSVC portability

### 2026-08-16: complete GPU-only reliability work — current status

The strict Vulkan executor now builds a second, purpose-specific SPIR-V module
for the medium detector's final `896x5x22` depthwise convolution.  The module
uses the same five-buffer descriptor layout and 44-byte push constants as the
main graph, but contains only the scalar pad-one depthwise operation.  Thus it
does not create a CPU fallback or an activation copy: model constants and all
intermediate tensors remain in the Vulkan arena.

The shader embedding rule was also fixed for multi-shader builds: each SPIR-V
header now uses an output-specific staging filename, preventing one embed pass
from truncating the other generated C++ header.

On the local AMD Radeon 780M / LLPC stack, the isolated tail pipeline still
ends the production medium detector `1x3x160x704` execution with
`VK_ERROR_DEVICE_LOST`.  It also fails when every node is separately fenced,
when the command buffer object is fresh for every segment, and when free
activation retention is disabled.  Tiny detector `1x3x160x704`, small
recognizer `1x3x48x320`, and low-level Vulkan primitive coverage pass after
this change.  Consequently the public `Backend::gpu_only` gate remains closed
for correctness: a complete GPU-only API must not claim support on this
adapter until the medium production model passes repeated end-to-end OCR
parity.  No CPU neural-network fallback is taken on failure.

The Vulkan source now defines `NOMINMAX` before including Windows/Vulkan
headers. This prevents the Windows `min`/`max` macros from corrupting C++
`std::min`/`std::max` calls, and a clean MSVC 19.51/Ninja Release build now
compiles the Vulkan backend and smoke tools. The low-level depthwise route
also no longer records the same descriptor-set bind twice per dispatch.

On AMD Radeon 780M, refreshed strict-graph checks (three warm passes, upload,
graph, and output download included) were:

| Model | Shape | Max absolute output error | CPU mean | Vulkan mean |
|---|---:|---:|---:|---:|
| tiny detector | `1x3x160x704` | `4.85918e-07` | 43.604 ms | 61.394 ms |
| small detector | `1x3x160x704` | `1.92900e-07` | 76.702 ms | 123.783 ms |
| medium detector | `1x3x128x704` | `7.23958e-09` | 156.387 ms | 627.358 ms |
| tiny recognizer | `1x3x48x320` | `4.70877e-06` | 12.895 ms | 54.201 ms |
| small recognizer | `1x3x48x320` | `8.94070e-07` | 26.585 ms | 153.499 ms |
| medium recognizer | `1x3x48x320` | `4.17233e-06` | 54.486 ms | 407.467 ms |

GPU RGB front-end plus strict graph validation passed for tiny and small
detectors on five varied PPM pages (`en`, `zh`, mixed, English-line and
Chinese-line), spanning `64x320`, `128x704`, and `192x704` detector inputs.
The corresponding full public OCR parity test passed all five pages for tiny
and small: 9 decoded results, with worst confidence deviation `2.9e-06` and
`1.7e-06` respectively. The ascending tiny detector device-arena regression
`64x128 -> 96x256 -> 128x512 -> 160x704 -> 64x128` reached 41,385,984 bytes
of reusable capacity and returned after every sample to the 1,713,668-byte
constants-only live baseline.

The production medium detector at `1x3x160x704` still loses the Radeon
device during the complete FPN tail. Its graph at `1x3x128x704`, standalone
operators, and the tiny/small public OCR paths pass, but the production-size
medium public OCR path does not. Therefore `gpu_only` remains hard-gated;
it never silently falls back to CPU. Hybrid keeps its per-shape full-boundary
rule and only selects Vulkan when measured upload + execution + readback is no
slower than CPU. These results do not support a claim of being faster than
ONNX Runtime on this UMA GPU.

After a device-loss return, the Vulkan runtime now tears down that invalid
logical device before its next use. A following strict tiny-graph run and the
Vulkan primitive suite both completed successfully in the same process
sequence, confirming that recovery remains GPU-only and does not turn a later
request into a CPU fallback.

### 2026-08-15: GPU-only allocator/recovery hardening

The strict Vulkan arena now exposes a monotonically changing resource
generation. `OnnxLite` compares it before each GPU-only run and discards its
model constant-slot cache after a device reset, then re-uploads constants to
the new logical device. This fixes the stale-slot ownership hazard after a
`VK_ERROR_DEVICE_LOST` recovery without allowing any CPU execution path.

Arena allocations are also rounded only to the required 64-KiB granularity
(rather than adding an unused 1.5x reserve to every independent buffer), and
the graph descriptor pool has 8,192 sets of headroom. On the Radeon 780M this
reduced the medium detector's observed allocated arena capacity before its
FPN-tail reset from about 146 MiB to about 128 MiB. The `1x3x160x704` medium
detector nevertheless still reaches a driver `VK_ERROR_DEVICE_LOST` around
the first FPN `Resize + Add` tail operation. Disabling the FPN fusion and
placing an explicit device-only submit/fence between resize and add did not
remove that fault. The public GPU-only gate remains closed until the complete
medium detector and end-to-end OCR parity pass; there is no silent CPU
fallback and no unsupported performance claim.

Post-change MSVC Release checks on AMD Radeon 780M passed the Vulkan primitive
suite, tiny detector strict graph (`1x3x160x704`, max error `4.85918e-07`,
GPU `52.2832 ms`) and small recognizer strict graph (`1x3x48x320`, max error
`8.9407e-07`, GPU `96.6479 ms`). Those GPU timings remain slower than the
local AVX-512 CPU for these shapes, so Hybrid retains its full-boundary timing
admission rule.

### 2026-08-15: GPU-only medium-tail qualification

The strict Vulkan executor now has an explicitly exercised scalar FP32
depthwise dispatch for the high-channel `896x5x22` tail shape.  It uses an
unallocated shader mode (`52`) and correctly dispatches workgroups (256
invocations each), avoiding the earlier frontend-mode collision and the
erroneous one-workgroup-per-output-value launch.  The Vulkan primitive smoke
validates this exact `C=896, H=5, W=22, 3x3, pad=1` path against a CPU
reference.  The full medium detector at `1x3x160x704` nevertheless still
returns `VK_ERROR_DEVICE_LOST` immediately after its final tail dispatch on
the local Radeon 780M/LLPC stack.  Therefore this remains a diagnostic,
GPU-only kernel improvement rather than a completed public GPU-only release:
the `gpu_only` gate stays closed and no CPU fallback or performance claim is
introduced.

### 2026-08-15: device-local Vulkan arena experiment

The Vulkan tensor arena now supports a device-local storage-buffer route with
host-visible staging buffers. `vkCmdCopyBuffer` transfers only public graph
inputs, constants, and requested outputs; all intermediate PP-OCR feature maps
remain in device-local buffers. Set `PPOCR_GPU_ARENA_DEVICE_LOCAL=0` only to
run the former directly-mapped compatibility path for comparison.

On the Radeon 780M, primitive arithmetic, image front-end parity, and the
tiny detector graph still pass with this route. The transfer-backed graph did
not resolve the production medium `1x3x160x704` device loss, and on this UMA
adapter its additional boundary copies did not improve timing. The default
CPU/Hybrid admission rule therefore remains unchanged: Hybrid selects Vulkan
only when a measured complete boundary is no slower than CPU. No ONNX Runtime
superiority is claimed from this experiment.

### 2026-08-15: multi-size GPU graph and residency recheck

The graph smoke matrix now covers detector inputs from `64x128` through
`192x704`, plus recognizer widths `64` and `320`, across all tiny/small/medium
families. On the Radeon 780M, all listed runs had device/CPU output parity:

| Model | Shape | CPU ms | Vulkan ms | Max abs. error |
|---|---:|---:|---:|---:|
| tiny detector | `1x3x64x128` | 5.675 | 16.897 | `1.13e-09` |
| tiny detector | `1x3x192x704` | 22.467 | 55.239 | `2.66e-05` |
| small detector | `1x3x64x128` | 10.150 | 38.459 | `7.74e-09` |
| small detector | `1x3x192x704` | 41.085 | 99.252 | `4.60e-07` |
| medium detector | `1x3x64x128` | 24.039 | 75.144 | `4.22e-10` |
| medium detector | `1x3x128x704` | 132.380 | 477.800 | `7.24e-09` |
| tiny recognizer | `1x3x48x64` | 5.274 | 16.755 | `3.52e-06` |
| small recognizer | `1x3x48x320` | 20.938 | 82.170 | `8.94e-07` |
| medium recognizer | `1x3x48x320` | 46.774 | 351.068 | `4.17e-06` |

The ascending tiny detector memory soak
`64x128 -> 96x256 -> 128x512 -> 160x704 -> 64x128` ended every sample at the
same constants-only live baseline (`1,713,668` bytes), proving activation
release after the largest input. `ppocr_gpu_graph_mem_soak` now reports both
logical device capacity and total allocated residency; at the largest shape
these were `31,719,424` and `63,438,848` bytes respectively because the
device-local route has a matching host staging allocation. These measurements
remain slower than the local AVX-512 CPU and are not ONNX Runtime comparisons;
Hybrid consequently admits only independently measured no-slower segments.

### 2026-08-16: GPU-only full-path requalification

The strict Vulkan path remains the intended implementation for complete
GPU-only OCR: image resize/normalization, detector and recognizer graphs, and
terminal CTC Top-1 execute on Vulkan storage. Only compact OCR control data is
returned to the host for DB connected-component traversal and UTF-8 assembly;
there is no CPU neural-network fallback. `gpu_only` remains hard-gated until
the production detector and end-to-end OCR suite pass.

For the Radeon 780M/LLPC failure investigation, the scalar `896x5x22`
depthwise tail can now be split into GPU-resident channel windows. The normal
path uses four windows; `PPOCR_GPU_TAIL_SLICES=1..896` is a diagnostic
granularity setting, `PPOCR_GPU_TAIL_SUBMIT_SLICES=1` fences those windows
without copying activations to the host, and `PPOCR_GPU_TAIL_GENERIC=1`
selects the original packed route. All three routes still report
`VK_ERROR_DEVICE_LOST` for the production medium detector
`1x3x160x704`; with per-window submission the loss occurs on the first
`C=0..223` scalar window. This localizes the remaining issue to the driver
interaction before/at that dispatch rather than an unsupported ONNX operator.

Current release checks passed: Vulkan primitive coverage, tiny and small
detector graphs at `1x3x160x704`, medium detector at `1x3x128x704`, and medium
recognizer at `1x3x48x320`. Measured one-run timings on this shared host were
`24.018/50.106 ms`, `37.939/86.509 ms`, `121.056/487.819 ms`, and
`51.734/358.526 ms` (CPU/Vulkan); maximum output errors were respectively
`4.86e-07`, `1.93e-07`, `7.24e-09`, and `4.17e-06`. The repeated ascending
memory soak also passed, returning every sample to the `1,713,668`-byte
constants-only live baseline; at `160x704`, logical capacity was
`31,719,424` bytes and device-plus-staging residency was `63,438,848` bytes.
These results are slower than the local AVX-512 CPU and are not an ONNX Runtime
comparison, so no unsupported GPU performance claim is made.

### 2026-08-16: pointwise-GELU Vulkan fusion qualification

The Vulkan shader now has a direct pointwise-convolution plus FP32-GELU
write mode. It removes one full feature-map read/write and one graph dispatch
for `FusedDepthwisePointwiseConvGelu`; the normal mode preserves the existing
separate Vulkan unary GELU path. `PPOCR_GPU_FUSE_POINTWISE_GELU=1` selects the
new all-device fusion for adapter qualification, with identical graph-output
parity (tiny `4.86e-07`, small `1.93e-07`, medium `7.24e-09`).

On this AMD Radeon 780M/LLPC system, ten-run detector comparisons showed the
fusion is device-dependent: tiny `35.466 -> 33.748 ms` (5.1% faster), small
`88.883 -> 91.514 ms` (3.0% slower), and medium `546.302 -> 643.987 ms`
(17.9% slower), normal/fused. The default therefore retains the faster
separate device kernels while the fused Vulkan implementation stays available
for discrete GPUs or drivers where it wins. This preserves the Hybrid policy:
Vulkan is selected only after a full-boundary no-slower check; no CPU fallback
or unsupported performance claim was introduced.

### 2026-08-16: complete GPU-only OCR qualification

`Backend::gpu_only` now executes the complete PP-OCRv6 neural path on Vulkan:
RGB resize/normalization, detector, recognizer, and CTC Top-1 remain on device.
Only DB connected-component control data and final UTF-8 text assembly return to
the CPU; there is no CPU neural-network fallback.  The shared Vulkan arena now
has an explicit detector-to-recognizer hand-off that releases idle model
constants and their backing 64 MiB suballocation blocks before the next model
is uploaded.  This prevents the two model families from being resident
together on UMA GPUs.

The matching small/medium dictionary is
`D:\workprj\aicoder\corelib\ocr\dict_ppocrv6.txt` (18,708 lines; 18,710
CTC classes); tiny uses `dict_ppocrv6_tiny.txt` (6,904 lines; 6,906 classes).
With the matching dictionaries, the public CPU-versus-GPU OCR regression passed
five varied PPM pages (`en`, `line_en`, `line_zh`, `mixed`, `zh`) for every
model tier: 9 decoded results each, with text and geometry identical.  Maximum
absolute confidence differences were `0.0494437` (tiny), `0.0136318` (small),
and `0.0081339` (medium). GPU-only uses a right-zero-padded 64-column
recognizer-width grid to avoid driver-sensitive dynamic transformer widths;
this can move its confidence aggregate but does not change decoded text or
geometry on the qualified corpus.

On the local AMD Radeon 780M / LLPC system, warmed strict-graph comparisons
(two runs; input/output boundary included) were:

| Model | Shape | CPU mean | Vulkan mean | Max abs. error |
|---|---:|---:|---:|---:|
| tiny detector | `1x3x160x704` | 22.356 ms | 45.910 ms | `4.85918e-07` |
| small detector | `1x3x160x704` | 35.894 ms | 99.619 ms | `1.92900e-07` |
| medium detector | `1x3x160x704` | 160.500 ms | 604.553 ms | `1.81899e-08` |
| tiny recognizer | `1x3x48x320` | 8.852 ms | 36.383 ms | `4.70877e-06` |
| small recognizer | `1x3x48x320` | 20.985 ms | 93.526 ms | `8.94070e-07` |
| medium recognizer | `1x3x48x320` | 51.310 ms | 370.130 ms | `4.17233e-06` |

The GPU memory soak `64x256 -> 96x512 -> 160x704 -> 64x256` passed. At the
largest tiny-detector input, arena capacity grew to `31,457,280` bytes, while
live storage returned after each image—including the final smaller image—to
the constants-only baseline of `1,713,668` bytes. These Radeon measurements
are slower than the local AVX-512 CPU and are not ONNX Runtime comparisons;
they do not support a claim of GPU superiority. Hybrid retains its
transfer-inclusive no-slower admission policy.

### 2026-08-16: GPU-only true batch scheduling

`RecognizeBatch` now has a GPU-only batch path. Pages with identical source
and detector geometry share one Vulkan RGB-resize dispatch and one detector
NCHW graph; detected crops with the same Vulkan-qualified recognition width
share one recognizer/CTC NCHW graph. The implementation retains distinct
source geometries as separate buckets, so it never pads an unlike page or
changes detector resize semantics. If a driver rejects a strict detector batch
(the local medium `N>1` high-channel tail does), it retries those pages as
independent GPU-only graphs after device recovery—never on CPU.

On four identical `720x152` `en.ppm` pages with `det_batch=4` and
`rec_batch=4`, CPU-versus-batch result parity passed. GPU-only timing on the
Radeon 780M was `434.004 ms -> 292.536 ms` serial/batched for tiny, a `1.484x`
throughput gain. The preceding small-model run measured `1480.410 ms ->
1249.591 ms` (`1.185x`). Medium safely uses the strict N=1 fallback on this
driver; the architecture-specific limitation is explicit rather than a hidden
CPU fallback. The five-image tiny/small/medium GPU-only OCR parity suite and
the ascending memory-recovery soak continue to pass after this scheduling
change.

### 2026-08-16: GPU-only single-page crop batching

The same Vulkan NCHW recognizer scheduler is now also used for repeated-width
text crops within a single `Recognize()` call. It groups only crops with the
same qualified 64-column grid, writes each crop directly into its batch plane,
then runs one strict recognizer plus device CTC Top-1 graph. Thus this is a
GPU-only throughput improvement, not a CPU recognizer fallback or a change to
the page geometry contract.

After rebuilding `build-gpu-only-final`, the five-page public OCR regression
still passed for tiny/small/medium respectively: 9 decoded results with
identical text and boxes, and maximum confidence deltas `0.0494437`,
`0.0136318`, and `0.0081339`. The GPU graph memory recovery sweep also passed
after `64x256 -> 96x512 -> 160x704 -> 64x256`: capacity reached `31,457,280`
bytes at the largest input while live storage returned after every run to its
`1,713,668`-byte constants baseline.

On the local Radeon 780M, a renewed warmed four-page `en.ppm` batch check
with `det_batch=4`, `rec_batch=4` measured tiny `530.291 -> 311.165 ms`
(`1.704x`) and small `1496.264 -> 1277.662 ms` (`1.171x`), serial/batched.
Both preserve exact GPU-only serial/batch output parity. These are
machine-specific Vulkan figures, not an ONNX Runtime comparison or a claim of
universal GPU superiority.

### 2026-08-16: GPU-only dense-page upload reuse and memory ladder

GPU-only recognizer batching now uploads each source page once per
same-width crop batch, rather than uploading the full RGB page once for every
text line. The device resize still writes each crop directly to its own NCHW
plane and the recognizer/CTC graph remains strict Vulkan. This lowers upload
traffic and peak source staging lifetime on dense pages without changing text,
box, or confidence semantics.

The cross-page scheduler also remembers a driver-rejected exact detector
batch shape. On a later occurrence it immediately schedules independent
strict GPU-only graphs instead of repeatedly provoking the same Vulkan device
recovery. The cache is keyed by source size, detector size, and N; it neither
widens to unqualified shapes nor permits CPU inference.

`bench/memory_ladder_regression.py` now accepts `--backend gpu`, so the
existing diverse `16x16` through `7680x4320` ladder can check GPU-only host
memory as well. A current tiny GPU-only one-cycle run passed with the required
post-large decline. Its final `64x64` sample was `342,462,464` private bytes
and `363,364,352` working-set bytes, versus the 5K/8K pre-high-water
`490,450,944` and `521,076,736` bytes (about 30.2%/30.3% lower). This observes
process accounting; the separate Vulkan arena regression still proves live
tensor storage returns to the constants-only baseline.

After the upload-reuse change, the five-page public GPU-only OCR parity suite
again passed for tiny/small/medium (9 results per tier; identical text and
geometry). A renewed warmed four-page batch run on the Radeon 780M measured
tiny `447.936 -> 295.321 ms` (`1.517x`) and small
`1542.215 -> 1280.809 ms` (`1.204x`), serial/batched. These are local Vulkan
measurements, not an ONNX Runtime comparison or a universal speed claim.

### 2026-08-16: measured Vulkan command-segment tuning

The strict Vulkan executor now defaults to 20 graph nodes per submitted
command segment (`PPOCR_GPU_ONLY_SEGMENT_NODES` remains an override). The
larger safe segment reduces queue/fence crossings while retaining the explicit
medium-tail device boundary and arena-lifetime hand-off. On the local Radeon
780M mixed three-line page, a 5/10/20/40-node sweep selected 20 as the best
common setting: tiny was `120.753 ms`, small `466.093 ms`, and medium
`2427.856 ms`; 40 nodes regressed all three and 32 regressed medium. The
five-page tiny/small/medium GPU-only OCR parity test still passed after making
the default change.

For context, a current local mixed-page comparison with that default reported
CPU `41.735 / 99.223 / 336.651 ms`, hybrid `52.127 / 124.219 / 531.973 ms`,
and strict GPU-only `122.299 / 479.389 / 2462.806 ms` for tiny/small/medium.
Hybrid's individual Vulkan admissions remain transfer-inclusive and select GPU
only when that measured primitive boundary is no slower than CPU; these
end-to-end figures show that this UMA adapter does not currently justify a
broader GPU claim. They are not an ONNX Runtime comparison.

### 2026-08-16: SIMD GPU-upload boundary coverage

The GPU-only RGB upload boundary now uses a runtime-dispatched byte-to-FP32
conversion: AVX-512 when the OS/CPU state enables it, AVX2 on compatible x86,
NEON on ARM, and scalar otherwise. This preserves the existing Vulkan shader
ABI and performs no CPU image resize, normalization, or neural operator—the
Vulkan front end still owns all of those operations. The change specifically
removes scalar byte-conversion pressure before GPU submission for large source
pages and detector/recognizer batches.

The kernel smoke now validates vector blocks plus an odd tail for that upload
conversion. Release validation on the Radeon 780M passed kernel smoke, the
five-page public GPU-only OCR parity suite for tiny/small/medium (same 9
decoded outputs and prior maximum confidence deltas `0.0494437`, `0.0136318`,
and `0.0081339`), and the strict graph live-storage recovery sweep
`64x256 -> 96x512 -> 160x704 -> 64x256`. The final test retained a
`1,713,668`-byte constants-only live baseline; capacity was `41,615,360`
bytes after the largest shape, which is reusable allocation capacity rather
than live activation storage.

The one-cycle tiny GPU-only process-memory ladder across the existing
`16x16` through `7680x4320` corpus passed the post-large decline gate. Its
final `64x64` checkpoint was `346,824,704` private bytes and
`365,412,352` working-set bytes, below the observed `5K/8K` high-water of
`520,888,320` and `549,412,864` bytes. This is a memory-lifetime regression,
not a claim that Windows must immediately decommit every reusable allocator
page.

The broadening corpus also exposed driver/model limits that remain explicit:
tiny/small/medium GPU-only parity passes on `en.ppm` and `mixed.ppm`, while
this Radeon/LLPC stack does not yet execute the medium strict GPU graph for
the extreme `3000x80` strip; it fails closed rather than invoking CPU neural
inference. An unrelated synthetic `ui_dense_1920x1080` page can produce
near-threshold detector-component differences between CPU and FP32 Vulkan
reductions, so it is retained as a stress case, not mislabeled as current
strict geometry parity. Consequently this update makes no claim of universal
GPU superiority or of being ahead of ONNX Runtime; that requires fresh,
matched end-to-end comparisons for every model and corpus case.

### 2026-08-16: strict GPU-only extreme-short detector qualification

The dedicated Vulkan depthwise-tail module now also qualifies the medium
detector's named terminal `896x1x30` shape, reached by the `3000x80` strip
after detector resizing to `32x960`. Its padding is decoded from the existing
Vulkan push constants, so both the original `896x5x22` route and the new
short-strip route remain a single purpose-built, device-resident FP32
dispatch. No ONNX Runtime or CPU neural fallback is introduced.

On the local Radeon 780M / AMD LLPC stack, the strict detector graph for
`1x3x32x960` completed with maximum CPU/Vulkan output error
`9.60426e-10`. The complete public GPU-only OCR check for medium on
`wide_strip_3000x80.ppm` also passed (one result; confidence delta
`0.0046322`). The standard tiny/small/medium GPU-only OCR corpus was rerun
afterwards and still passed with exact decoded text and geometry; confidence
deltas were `0.0494437`, `0.0136318`, and `0.0081339` respectively.

The extension deliberately remains narrowly qualified. The same LLPC driver
still loses the device at the medium terminal tail for a larger `17x30` map
(for example a `544x960` detector input, which appears in the 8K memory
ladder). GPU-only continues to fail closed for that unsupported shape. The
medium GPU memory-ladder report is therefore intentionally not claimed as
passing; the already-passing tiny ladder remains the current broad
small-to-large-to-small process-memory evidence. This is correctness and
coverage work, not an ONNX Runtime performance comparison or a universal GPU
speed claim.

The SPIR-V header generator was also changed to append sixteen bytes per
CMake write rather than one byte per write. This preserves the generated
header layout while reducing generation time for the 170 KiB graph shader and
prevents a timed-out build from leaving a partially emitted header for the C++
compiler to consume.

### 2026-08-16: strict GPU-only CTC confidence correction

The terminal GPU CTC Top-1 reduction now returns the selected value from the
model's already-computed Softmax tensor.  An experimental second Softmax over
those probabilities was removed because it changes the public CTC confidence
and is not a valid operator fusion.  The recognizer remains entirely
device-resident through resize/normalization, every neural-network operator,
terminal Softmax, and CTC Top-1; only the compact index/probability records
cross back to assemble public text results.

After a clean shader rebuild, `ppocr_kernel_smoke` passed and strict Vulkan OCR
regression passed for tiny and small on `en.ppm`, `line_en.ppm`, and
`mixed.ppm`, with maximum CPU/GPU confidence deltas `0.0494437` and
`0.0136318`. Medium passed the same three images plus
`wide_strip_3000x80.ppm` with a maximum confidence delta of `0.0081339`.
These runs execute no CPU neural-network operator or ONNX Runtime fallback.
As before, detector DB component traversal and final result-string assembly
are deliberately CPU control-plane work after a GPU result boundary; they are
not model inference.

### 2026-08-16: terminal Softmax-to-CTC Vulkan fusion and broad memory check

For the strict GPU-only recognizer, the terminal `Softmax -> CTC Top-1`
sequence is now fused without materialising the full terminal probability
tensor. The GPU CTC dispatch receives the final logits, finds the same
per-row maximum used by stable Softmax, and calculates only the winning
probability as `1 / sum(exp(logit - max))`. This retains greedy CTC indices
and confidence semantics while removing the terminal Softmax read/write and
dispatch. Ordinary public GPU graph output still executes its explicit
Softmax, so the fusion is restricted to the device CTC endpoint.

The three-tier GPU-only OCR regression was rebuilt after the fusion. Tiny and
small passed `en.ppm`, `line_en.ppm`, and `mixed.ppm`; medium passed those
three plus the `3000x80` short strip. Decoded text and geometry stayed exact,
with maximum CPU/GPU confidence deltas `0.0494437`, `0.0136318`, and
`0.0081339`. `ppocr_kernel_smoke` also passed.

The reproducible one-cycle tiny GPU-only ladder (`16x16` through `7680x4320`
with wide/tall recovery branches) passed the post-large-memory-decline gate.
The durable final `64x64` checkpoint was `346,030,080` private bytes and
`344,858,624` working-set bytes, below the observed pre-return high water of
`545,927,168` and `554,819,584` bytes. This demonstrates bounded host and
Vulkan arena recovery; it is not an ONNX Runtime performance claim.

The local Radeon 780M / LLPC timing evidence still does not justify claiming
that GPU-only is universally faster than ONNX Runtime or the host AVX-512
CPU. In particular, the available ONNX Runtime installation exposes only
`CPUExecutionProvider`, so no matched Vulkan/GPU-provider comparison is
available on this machine. Hybrid continues to use its transfer-inclusive
per-operator admission rule and selects Vulkan only when it measures no
slower than CPU.

### 2026-08-16: reproducible native-build ORT comparison

`bench/compare.ps1` and `bench/compare_models.ps1` now accept `-NativeExe`.
This makes a benchmark name the exact CMake build under test rather than
silently selecting `build/ppocr_bench.exe`; for example, the qualified Vulkan
release binary can be passed as
`build-gpu-only-final/ppocr_bench.exe`. The path is emitted in the JSON report
for auditability.

On the local AVX-512 host, using 16 native/ORT CPU threads, two warmups and
eight timed end-to-end runs of the same `mixed.ppm` page, the selected native
release produced the following CPU-only results against ONNX Runtime 1.28.0
`CPUExecutionProvider`:

| PP-OCRv6 model | native mean | ORT mean | native latency improvement |
| --- | ---: | ---: | ---: |
| tiny | 58.679 ms | 164.644 ms | 64.36% |
| small | 100.198 ms | 251.448 ms | 60.15% |
| medium | 433.107 ms | 760.844 ms | 43.08% |

This is a matched local CPU comparison: detector preprocessing, DB postprocess,
recognizer preprocessing and CTC decode are included by both harnesses. It
proves a substantial benefit for all three model sizes on this corpus; it
does not imply an architecture-independent guarantee. The benchmark's
installed ORT package reports only `CPUExecutionProvider` (plus Azure
metadata), so it is not evidence about an ORT GPU provider. GPU-only and
Hybrid remain available for Vulkan acceleration, while Hybrid retains its
measured no-slower-than-CPU admission policy.

### 2026-08-16: adaptive Vulkan CTC reduction experiment

The Vulkan CTC Top-1 shader now contains a cooperative 256-lane vocabulary
reduction for adapters where the terminal recognizer rows are a GPU bottleneck.
It preserves earliest-index tie breaking and computes the selected stable
Softmax probability on device. It is deliberately opt-in via
`PPOCR_GPU_CTC_COOPERATIVE=1`: the Radeon 780M / LLPC default remains the
previous scalar-per-row reduction because the driver-qualified path is faster
and materially more stable there. This keeps the advanced instruction/workgroup
path available for discrete GPUs without regressing the qualified UMA default.

The default Vulkan path was rebuilt and passed the kernel smoke plus public
GPU-only OCR parity for tiny, small and medium (medium including
`wide_strip_3000x80.ppm`). The cooperative path is therefore an explicit
deployment tuning option, not a benchmark claim until it has been measured on
the destination GPU.

### 2026-08-16: GPU-only full-network contract revalidation

The public `Backend::gpu_only` route remains a strict Vulkan neural-network
execution path: RGB resize/normalization, detector, recognizer transformer,
terminal Softmax/CTC Top-1, and recognizer crop batching execute on Vulkan;
construction or execution fails closed if the complete model graph cannot be
lowered. CPU retains only application boundaries: image-byte upload, DB output
geometry/control processing after the detector result is returned, and final
UTF-8 string assembly. No ONNX Runtime library or CPU neural operator is used
by this route.

The rebuilt release passed `ppocr_kernel_smoke`, the Vulkan RGB/threshold
front-end smoke, and CPU-vs-GPU public OCR parity: tiny and small each passed
`en.ppm`, `line_en.ppm`, and `mixed.ppm`; medium passed those plus
`wide_strip_3000x80.ppm`. Maximum confidence deltas were respectively
`0.0494437`, `0.0136318`, and `0.0081339`, while decoded text and geometry
were exact.

A fresh local Radeon 780M / LLPC steady-state tiny GPU-only probe on
`mixed.ppm` (one warm-up, three timed runs) measured `137.016 ms` mean,
`131.292 ms` median, and `7.298 FPS`. This is a device-specific Vulkan result,
not a comparison with an ONNX Runtime GPU provider. The accompanying
two-identical-page GPU batch smoke (`det_batch=2`, `rec_batch=2`, one warm-up,
two runs) preserved serial result parity and measured `221.428 ms` serial vs
`194.733 ms` batched mean (`1.137x`).

### 2026-08-16: GPU-only small-model immutable-residency cache

GPU-only OCR now keeps a detector/recognizer pair's fused immutable Vulkan
weights resident across the detector-to-recognizer hand-off when their combined
payload is at most `64 MiB` (override with `PPOCR_GPU_CONSTANT_CACHE_MB` before
constructing `OCR`). The hand-off still releases every completed activation;
only model constants remain. Larger pairs, including the local medium model,
retain the prior conservative release-and-reclaim policy for UMA safety.

This changes no inference operator or fallback policy. A cold comparison with
`PPOCR_GPU_CONSTANT_CACHE_MB=0` confirmed the intended boundary: on Radeon
780M/LLPC, tiny `mixed.ppm` with two warm-ups/eight runs changed from
`318.372 ms` (re-upload per stage) to `133.153 ms` with the resident pair
(`2.39x`). The resident run retained `128.241 ms` median and `7.510 FPS`.
Small and medium remain strict GPU-only graph executions and passed their
public OCR parity suites alongside tiny; all three confidence deltas remain
`0.0494437`, `0.0136318`, and `0.0081339` respectively.

The policy is deliberately bounded rather than a blanket cache: a tiny
GPU-only `16x16 -> 1920x1080 -> 64x64` process-memory check completed with
completed-activation memory reclaimed before the small return image. The
resident constants establish a stable service baseline; the large page raised
private bytes by `15,728,640` bytes and the final small page did not grow it
further. A full ascending-to-8K ladder remains the stronger regression gate.

### 2026-08-16: cached GPU-only 8K memory ladder and three-mode snapshot

The full tiny GPU-only memory regression was rerun after enabling bounded
immutable-pair residency, using the existing `16x16` through `7680x4320`
resolution/aspect-ratio ladder and its recovery tail. It passed
`--require-post-large-decline`. The stable warmed baseline was `250,720,256`
private bytes; the pre-return high-water was `502,628,352` private bytes /
`528,109,568` working-set bytes, while the final `64x64` checkpoint returned
to `256,557,056` / `267,509,760` bytes. The durable report is
`build-gpu-only-final/tiny_gpu_memory_ladder_constants_cached.json`.

For a separate steady-state `mixed.ppm` snapshot (16 native worker threads,
two warm-ups, eight timed runs), the same release measured:

| model | CPU-only mean | Hybrid mean | GPU-only mean | preferred local mode |
| --- | ---: | ---: | ---: | --- |
| tiny | 45.476 ms | 53.561 ms | 133.153 ms | CPU-only |
| small | 93.684 ms | 170.108 ms | 462.873 ms | CPU-only |
| medium | 376.173 ms | 542.679 ms | 2435.894 ms | CPU-only |

These figures are intentionally not presented as a universal GPU win: the
Radeon 780M is an UMA adapter and this full page is CPU-SIMD-favourable.
Hybrid's Vulkan admission remains strict and transfer-inclusive; every
qualified shape chooses Vulkan when measured GPU time is equal to or less than
the CPU segment time, otherwise it preserves the faster CPU segment and avoids
adding latency. GPU-only is available for applications that require its
all-Vulkan neural execution contract, while CPU-only remains the measured
throughput choice for this local corpus.

### 2026-08-16: GPU-only contract recheck

The release build was rechecked after the GPU-only integration pass.  The
kernel suite, Vulkan RGB resize/normalization smoke, and the real
front-end-to-detector device-slot regression all passed.  On `en.ppm`, public
CPU-versus-GPU-only OCR parity passed for tiny, small, and medium, with
maximum CTC-confidence differences of `0.0494437`, `0.0123513`, and
`0.0081339`; decoded text and box geometry were exact.  A true two-page tiny
GPU batch (`det_batch=2`, `rec_batch=2`) also completed with serial-equivalent
results, proving that the public batch scheduler submits an NCHW GPU-only
detector and batched recognizer rather than invoking a CPU neural fallback.

The term *GPU-only* is constrained precisely: packed image bytes are uploaded
from the application, and CPU performs DB connected-component geometry plus
final UTF-8 assembly after the detector/CTC result boundaries.  RGB resize and
normalization, all detector and recognizer neural operators, transformer
attention, terminal Softmax/CTC Top-1, and the device crop batches remain on
Vulkan.  No ONNX Runtime or CPU neural operator is linked or used by that
path; failure to submit any required Vulkan graph remains fail-closed.

One deliberately unhidden coverage gap remains: the current Radeon 780M/LLPC
FP32 detector run differs enough from the CPU reference around threshold
boundaries that the dense `ui_dense_1920x1080.ppm` page does not yet preserve
all CPU box/text records.  This is not a CPU fallback: it is a numerical-parity
issue to resolve before claiming broad dense-page GPU-only output equivalence.
The qualified normal-page and batch contract above remains passing.

### 2026-08-16: GPU-only DB postprocess completion

`Backend::gpu_only` now retains the detector probability map on Vulkan through
the full DB postprocess: thresholding, 8-connected-component traversal, mean
score filtering, rectangular unclip, source-coordinate conversion, and reading
order.  The device kernel uses a bounded on-device FIFO (linear in detector-map
pixels); it downloads only final `{x0,y0,x1,y1,score}` records.  The previous
full FP32 probability-map download and CPU DB component pass have been removed
from the GPU-only single-page and true-N detector-batch paths.  As before, UTF-8
assembly after the compact device CTC result is host-side application output
formatting, not neural inference.

Rebuilt `build-gpu-only-final` validation on AMD Radeon 780M/LLPC passed:

| Check | Coverage | Result |
| --- | --- | --- |
| Kernel + Vulkan front-end | SIMD kernels; RGB resize/normalize; GPU threshold | passed |
| GPU-only OCR, tiny | `en`, `line_en`, `mixed` | 6 results, max confidence delta `0.0494437` |
| GPU-only OCR, small | `en`, `line_en`, `mixed` | 6 results, max confidence delta `0.0136318` |
| GPU-only OCR, medium | `en` | 2 results, max confidence delta `0.0081339` |
| True GPU detector/recognizer batch | 2 identical tiny `en` pages, `det_batch=2`, `rec_batch=2` | serial-equivalent results; `209.975 -> 193.990 ms` (`1.082x`) |

This is a real GPU-only execution contract—no ONNX Runtime and no CPU neural or
DB floating-point fallback are linked into the route.  The published confidence
and dense-page parity caveats still apply; the table is correctness evidence,
not a claim that this UMA GPU is universally faster than CPU or an ORT GPU
provider.

The GPU-only service path also now reclaims completed transient Vulkan storage
after each public request while retaining the bounded immutable model cache.
`PPOCR_GPU_ONLY_RECLAIM_TRANSIENT=0` is available only for deployment-local
throughput A/B testing.  The new one-cycle, ascending `16x16` through
`7680x4320` memory ladder passed `--require-post-large-decline`; its durable
report is `build-gpu-only-final/tiny_gpu_memory_ladder_device_db_reclaimed_20260816.json`.
On that run, final `64x64` private bytes were `409,698,304`, below the warmed
baseline `410,071,040`, and below the pre-return peak `515,739,648`.

### 2026-08-16: native-byte GPU-only image ingress

The strict Vulkan front end now uploads the application RGB buffer as packed
native bytes.  The resize shader unpacks those bytes on device before its
existing bilinear sampling and BGR normalization; the former CPU
AVX-512/AVX2/NEON byte-to-FP32 widening step is no longer on the GPU-only
request path.  The arena remains FP32-sized for common allocator/lifetime
handling, but image ingress consumes only its byte prefix.  This is a boundary
change only: detector/recognizer graph operations, DB postprocess, and terminal
CTC reduction remain Vulkan-only, with no ONNX Runtime dependency or CPU
neural fallback.

Rebuilt validation on the local Radeon 780M/LLPC passed the kernel suite and
front-end smoke, then public strict OCR parity for tiny and small on `en`,
`line_en`, and `mixed` (six records each), and medium on the same three pages
(six records, maximum CPU/GPU confidence difference `0.0081339`). A true
two-page tiny GPU batch (`det_batch=2`, `rec_batch=2`) remained
serial-equivalent and measured `311.699 -> 227.434 ms` mean (`1.371x`) in a
three-run local probe. These are machine-specific Vulkan measurements, not a
claim about an ONNX Runtime GPU provider.

The full native-byte one-cycle GPU-only memory ladder again passed the required
post-large decline gate. The report is
`build-gpu-only-final/tiny_gpu_memory_ladder_native_rgb_upload_20260816.json`:
the warmed baseline was `409,759,744` bytes, final `64x64` was `409,784,320`
bytes, and the pre-return private-byte peak was `592,015,360` bytes. Thus the
completed large-image workspace fell by 30.8% before the final small-image
checkpoint.

### 2026-08-22: strict GPU-only requalification and CTC metadata correction

`Backend::gpu_only` remains a fail-closed Vulkan path: RGB upload/resize and
normalization, detector, Vulkan DB postprocess, crop resize, recognizer, and
CTC Top-1 run on the device. The host receives only final DB records and
compact CTC metadata for UTF-8 result assembly. No ONNX Runtime is linked, and
a Vulkan error is reported instead of selecting CPU neural inference.

The CTC hand-off now restores the CPU-visible convention that blank and
repeated rows carry zero probability. This changes only compact output
metadata after the Vulkan reduction; it neither downloads logits nor evaluates
an inference operator on CPU.

Fresh Radeon 780M / LLPC checks from `build-gpu-only-ep`:

| Check | Result |
| --- | --- |
| Tiny detector `1x3x160x704`, 2 runs | max error `5.63276e-7`; CPU/GPU mean `10.269 / 45.892 ms` |
| Small detector `1x3x160x704`, 2 runs | max error `3.05939e-7`; CPU/GPU mean `19.456 / 65.210 ms` |
| Medium detector `1x3x160x640`, 2 runs | max error `7.56991e-8`; CPU/GPU mean `80.544 / 266.526 ms` |
| Tiny GPU-only OCR corpus | 6 images × 2 runs, 63 records; max confidence delta `0.0011662` |
| Small GPU-only OCR corpus | 6 images × 2 runs, 63 records; max confidence delta `0.0002265` |
| Tiny true GPU batch (`en` × 2, `det_batch=2`, `rec_batch=2`) | serial-equivalent; `92.841 -> 83.830 ms` (`1.107x`) |

The corpus covers `en`, `mixed`, low-contrast, ultra-wide, ultra-tall, and
`1920x1080` screenshot pages. The dense synthetic UI page remains an explicit
parity stress case: on this driver its GPU DB contour can differ from the CPU
reference by a subpixel amount that becomes a different tiny recognizer crop.
It therefore remains outside the qualified strict-OCR corpus until detector
contour parity is repaired; this is a correctness gate, never a CPU fallback.
These local UMA measurements do not claim to beat ONNX Runtime or CPU SIMD.

### 2026-08-22: GPU-only execution and memory recheck

The current public `Backend::gpu_only` route is an all-Vulkan inference
contract: RGB upload/resize/normalization, detector, DB threshold/component
postprocess, recognizer crop resize, recognizer, and CTC Top-1 stay on the
device. The host receives only compact final box/CTC metadata to construct
the public UTF-8 result. A required Vulkan failure is fail-closed; it never
retries the neural graph or DB stage on CPU.

After rebuilding `build-gpu-only-ep`, the following local Radeon 780M / LLPC
checks passed:

| Check | Result |
| --- | --- |
| Kernel SIMD/Vulkan smoke | passed |
| Tiny detector `1x3x160x704` | max error `5.63276e-7`; CPU/GPU `10.816 / 44.449 ms` |
| Small detector `1x3x160x704` | max error `3.05939e-7`; CPU/GPU `40.614 / 78.777 ms` |
| Medium detector `1x3x160x640` | max error `7.56991e-8`; CPU/GPU `86.750 / 277.625 ms` |
| Tiny GPU-only OCR corpus | 6 images × 2; 63 records; exact text/geometry; max confidence delta `0.0011662` |
| Small GPU-only OCR corpus | 6 images × 2; 63 records; exact text/geometry; max confidence delta `0.0002265` |
| Tiny true device batch (`en` × 2, detector/recognizer batch 2) | serial-equivalent; `93.460 -> 85.286 ms` (`1.096x`) |

The same tiny instance was then warmed over an ascending source ladder
(`16x16`, `en`, `mixed`, low contrast, `3000x80`, `80x3000`, screenshot,
dense UI), followed by `en` again. The post-large `en` checkpoint was
`605,151,232 B` private / `505,942,016 B` working set, below the preceding
large-page high-water (`605,732,864 B` / `506,232,832 B`). This verifies that
completed large-image transient storage is released/reused rather than
retained as an ever-growing per-request allocation.

The synthetic dense UI corpus still has a known CPU/GPU result-parity failure
caused by a subpixel DB-component boundary becoming a different tiny crop.
It remains an explicit regression target, not a supported parity claim and
not a condition that permits CPU fallback. Production medium detector
`1x3x160x704` likewise remains driver-unqualified on this adapter and reports
`VK_ERROR_DEVICE_LOST` fail-closed. These UMA measurements are correctness
and memory-lifetime evidence; they do not claim a universal performance win
over ONNX Runtime or CPU SIMD.

### 2026-08-22: Vulkan admission and multi-mode recheck

Hybrid keeps its required transfer-inclusive admission rule: a Vulkan segment
is used only after the same complete boundary is measured no slower than the
CPU SIMD segment.  The optional complete-GPU-graph experiment is now read once
when `OCR` is constructed; it cannot accidentally supersede the normal
per-operator no-slower policy during a request.  This protects hybrid latency
and CPU utilization on UMA adapters, where a valid complete Vulkan graph can
still be slower than the CPU path.

On the local Radeon 780M / AVX-512 host, `mixed.ppm` (three text regions,
two warm-ups/five timed runs) measured the following C++ implementation
values:

| Model | CPU-only | Hybrid | GPU-only |
| --- | ---: | ---: | ---: |
| tiny | `15.616 ms` | `15.888 ms` | `60.750 ms` |
| small | `41.700 ms` | `42.537 ms` | `173.970 ms` |
| medium | `175.402 ms` | `201.552 ms` | unavailable on this production OCR shape |

Hybrid tracing for medium also observed two eligible Vulkan segments: crop
RGB+Conv.0 (`1.426` vs `1.632 ms`, GPU/CPU) and one terminal CTC GEMM
(`5.438` vs `5.636 ms`). A slower same-model CTC shape (`5.058` vs
`4.350 ms`) remained on CPU. This is the intended behavior: Vulkan lowers
CPU work when it is equal or faster, while slower local segments do not make
the request slower.

The revised strict Vulkan regression passed tiny and small six-image OCR
corpora (two repeats, 63 records each), the kernel smoke, and a true tiny
N=2 detector/recognizer batch (`98.783 -> 91.599 ms`, `1.078x`). The
small-to-large-to-small GPU memory ladder retained a bounded final `en`
checkpoint of `605,863,936 B` private / `507,125,760 B` working set after
the screenshot and dense-UI phases. These are local results, not an
unsubstantiated claim of universal superiority over ONNX Runtime; a matched
ORT benchmark on the target GPU remains required before making that claim.

### 2026-08-22: GPU-only replay result consistency

The strict GPU-only replay path now applies the same compact CTC metadata
convention as the non-replayed GPU path: blank and repeated CTC rows have
zero emitted-character probability. This is performed only after the Vulkan
CTC reduction has downloaded its `[N,T]` result records; logits are never
downloaded and no CPU neural operator is introduced. It prevents a page with
multiple replayed recognition-width graphs from reporting a confidence that
differs merely because its command buffers were coalesced into one queue
submission.

During the same requalification, the approximate AVX2/AVX-512 vector-exp
sigmoid was changed to an explicit deployment opt-in
(`PPOCR_ENABLE_AVX512_SIGMOID` or `PPOCR_ENABLE_AVX2_SIGMOID`). The default
uses the exact scalar-libm logistic expression because detector probabilities
feed the DB threshold boundary. This restores the tight kernel numerical
check while preserving optional ISA experimentation for deployments that have
qualified its FP32 drift.

Fresh release checks on the local AMD Radeon 780M / LLPC stack:

| Check | Result |
| --- | --- |
| Backend probe | Vulkan loader/compute/full graph: `1/1/1`; device `AMD Radeon 780M Graphics` |
| Kernel smoke | passed |
| Detector graph, tiny `1x3x160x704` | max error `5.63276e-7`; CPU/GPU mean `10.798 / 32.052 ms` |
| Detector graph, small `1x3x160x704` | max error `3.05939e-7`; CPU/GPU mean `20.878 / 51.629 ms` |
| Detector graph, medium `1x3x160x640` | max error `7.56991e-8`; CPU/GPU mean `103.525 / 290.357 ms` |
| GPU-only OCR, tiny | 7 images × 2 runs; 64 records; exact text/geometry; max confidence delta `0.0011663` |
| GPU-only OCR, small | 7 images × 2 runs; 64 records; exact text/geometry; max confidence delta `0.0002261` |
| GPU-only replay stress, tiny screenshot (`rec_batch=2`) | 3 runs; 34 records; exact text/geometry; max confidence delta `0.0003594` |

The qualified image set is `16x16`, English, mixed-script, low-contrast,
ultra-wide, ultra-tall and `1920x1080` screenshot pages. The production
medium OCR detector shape remains fail-closed on this Radeon/LLPC driver with
`VK_ERROR_DEVICE_LOST`; it is a driver qualification limitation, not a CPU
fallback. These results establish complete GPU-only execution and replay
consistency, but they are not a claim that this UMA Vulkan implementation
outperforms ONNX Runtime.

### 2026-08-22: persistent replay activation-release regression

The strict Vulkan replay cache now owns the full set of arena slots captured
while its command buffer is recorded, including private intermediate
activations and manually pinned RGB/input slots.  Dropping a replay graph (on
model release or optional cache eviction) unpins and releases this complete
set before reusable arena storage is reclaimed.  This closes a device-memory
lifetime gap where only public input/output descriptors were previously
released.

`ppocr_gpu_graph_mem_soak` now accepts `--device-input-replay`.  Unlike the
ordinary host-input memory ladder, this mode forces persistent command-buffer
record/replay, releases the model after every shape, and requires both
`live_bytes` and `allocated_bytes` to be zero at each `replay-release`
checkpoint.  It is a logical Vulkan-arena test; Windows process
`PrivateUsage` can still retain allocator high-water pages and must not be
misreported as immediately returned physical memory.

Validated with the release build on AMD Radeon 780M / LLPC:

| Model | Device-input replay ladder | Result |
| --- | --- | --- |
| tiny detector | `1x3x64x128 -> 96x256 -> 128x512 -> 64x128` | every release: `live=0`, `allocated=0` |
| small detector | `1x3x64x128 -> 96x256 -> 128x512 -> 64x128` | every release: `live=0`, `allocated=0` |

The normal small-to-large-to-small host-input ladder also remains green:
tiny reaches a `46,006,272`-byte reusable arena high-water and returns after
every sample to its `1,713,924`-byte immutable baseline; small reaches
`73,007,104` bytes and returns to `9,813,844` bytes.  Capacity is deliberately
reusable, while the immutable baseline proves no dynamic activation stays
live after the largest shape.

Reproduce the replay lifetime test:

```powershell
$models = 'D:\workprj\aicoder\.tmp\ocr-models'
.\build-gpu-only-ep\ppocr_gpu_graph_mem_soak.exe `
  "$models\ppocrv6_tiny_det.onnx" --device-input-replay `
  1 3 64 128 1 3 96 256 1 3 128 512 1 3 64 128
```

The revised batch fallback keeps the GPU-only contract strict: a rejected
N-batch is remembered for that exact shape and then retried as independent
N=1 Vulkan pages.  It does not attempt a speculative runtime reset and never
falls through to CPU neural inference.  Current same-page N=2 checks remain
serial-equivalent: tiny `89.904 -> 84.510 ms` (`1.064x`) and small
`247.571 -> 236.356 ms` (`1.047x`).  Production medium at its normal OCR
shape is still driver-unqualified on this adapter and fails closed with
`VK_ERROR_DEVICE_LOST`; no successful medium GPU-batch claim is made.

### 2026-08-22: three-size diverse CPU/ONNX Runtime sample

`bench/compare_models.ps1` now defaults to a broader real-image set: tiny
input, single word, English, Chinese, mixed-script, low contrast, wide/tall
strips, and `1920x1080`/`2560x1440` screenshots.  This makes the standard
benchmark cover scale and aspect-ratio changes instead of only a short
English line.

For a quick, reproducible CPUExecutionProvider comparison, the release build
was measured with one warm-up and one sample each on `en`, `mixed`, and
`3000x80` wide-strip pages (16 native workers, ONNX Runtime 1.28.0 CPU EP,
same detector/recognizer ONNX files and pipeline semantics).  The captured
JSON is `build-gpu-only-ep/cpu_ort_sampled_20260822.json`.

| Model | Native total | ORT total | Latency improvement | Worst sampled case |
| --- | ---: | ---: | ---: | ---: |
| tiny | `45.357 ms` | `301.243 ms` | `84.94%` | `81.02%` |
| small | `116.614 ms` | `494.681 ms` | `76.43%` | `65.35%` |
| medium | `475.979 ms` | `842.178 ms` | `43.48%` | `16.34%` |

These are CPU-only results, so they demonstrate the native SIMD/fused path
against the available matched ORT CPU provider rather than incorrectly
attributing the result to Vulkan.  The short single-sample run is useful as
a cross-model sanity check, not a replacement for a longer deployment
benchmark.  GPU-only tiny/small and hybrid retain their independently tested
Vulkan policy; medium GPU-only continues to fail closed on the tested
Radeon/LLPC production shape.

### 2026-08-22: hybrid admission and GPU residency tightening

Hybrid admission now uses one strict rule for its RGB preprocessing, fused
detector-stem, recognizer-stem and graph-level Vulkan boundaries: measured
upload + dispatch + readback must be `GPU <= CPU`.  The earlier 15% latency
grace for three RGB/stem paths was removed, so hybrid cannot trade a slower
request merely to move work from CPU.  Equal-or-faster GPU segments are still
preferred, exactly as the public backend contract requires.

On the Radeon 780M / LLPC trace, the small mixed page correctly rejected the
slower GPU detector stem (`8.360 / 6.009 ms`), recognizer stem
(`1.130 / 0.892 ms`) and CTC projections.  A medium `1920x1080` trace also
admitted only qualifying CTC shapes (for example `15.470 / 21.808 ms`), while
leaving slower shapes on CPU.  Thus hybrid lowers CPU use only where the full
boundary is non-regressive; it is not a blanket Vulkan offload mode.

`PPOCR_GPU_RECLAIM_TRANSIENT=0|1` is now the common retention switch for both
GPU-only and optional complete-graph hybrid execution; the old
`PPOCR_GPU_ONLY_RECLAIM_TRANSIENT` spelling remains supported.  The default
is `1`.  A GPU-only tiny multi-size ladder (`16x16`, normal, wide, tall,
1080p, 1440p, then `16x16`) has been captured for the default and explicit
zero-retention policies in `build-gpu-only-ep/tiny_gpu_memory_diverse_20260822.json`
and `build-gpu-only-ep/tiny_gpu_memory_diverse_retain0_20260822.json`.
Windows process accounting still retains allocator high-water pages, so the
authoritative per-graph lifetime gate remains the Vulkan arena's
`live_bytes`/`allocated_bytes` replay regression rather than PrivateUsage
alone.

### 2026-08-22: runtime-generation-safe hybrid admission

The RGB/stem hybrid admission caches are now keyed by the Vulkan runtime
generation as well as image and operator shape.  A device-loss recovery tears
down arena buffers and can select a different physical-device configuration;
reusing a pre-recovery yes/no verdict would be unsound.  The next request now
re-runs its transfer-inclusive qualification automatically, while normal
same-generation requests remain a fast cache lookup.

An AVX-512 sigmoid A/B was also rechecked.  On the 16-worker English sample
it did not show a stable all-model speedup (tiny `14.769 ms`, small
`33.743 ms`, medium `145.255 ms`), so it stays opt-in.  Tiny and small
eight-image GPU-only parity runs still passed when the CPU-side option was
present (exact text/geometry; confidence deltas `0.0011663` and `0.0002261`).
Keeping exact default detector sigmoid protects DB threshold boundaries while
leaving a qualified deployment an explicit SIMD option.
