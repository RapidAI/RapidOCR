# CPU performance

The hot path is the built-in ONNX interpreter: convolutions, GEMM, and the
RGB-to-NCHW front end. Post-process (DB boxes, CTC decode) is small next to
those kernels. Threads come from a persistent pool (`PPOCR_BENCH_THREADS`,
default `min(16, hardware_concurrency)`) plus the recognizer's outer batch
pool, which is capped by the hardware thread count.

## One binary, three targets

| Target | Where the code lives | When it runs |
| --- | --- | --- |
| x86 without AVX (scalar / SSE2) | `src/kernels.cpp`, compiled with `-mno-avx -mno-avx2 -mno-avx512f` | no AVX2 |
| x86 AVX2 + FMA, no AVX-512 | `src/kernels_avx2.cpp`, compiled with `-mavx2 -mfma -mno-avx512f -mno-avx512bw -mno-avx512vl -mno-avx512dq` | `avx2` and `fma` CPUID bits, and `PPOCR_FORCE_ISA` is not `scalar` |
| x86 AVX-512 | `src/kernels_avx512.cpp`, compiled with `-mavx512f -mavx512dq -mavx512bw -mavx512vl -mbmi -mbmi2 -mfma` | AVX-512F/DQ/BW/VL plus AVX2, FMA, BMI1, and BMI2, and OS XSAVE state, unless `PPOCR_DISABLE_AVX512` or `PPOCR_FORCE_ISA=avx2`/`scalar` |
| AArch64 NEON | `src/kernels.cpp` (NEON is baseline on AArch64) | not forced off with `PPOCR_FORCE_ISA=scalar` |

An AVX2-only machine never executes the AVX-512 object file. `HasAvx512()`
returns false unless CPUID and XCR0 say the OS has enabled AVX-512, so
`PPOCR_FORCE_ISA=avx512` on that machine stays on AVX2 or scalar. The
`isa_encoding` test disassembles the baseline and AVX2 objects and fails if
they contain `zmm` or AVX-512 mask registers.

`PPOCR_FORCE_ISA=scalar|avx2|avx512|neon` selects a legal path for benchmarks.
It cannot turn on an ISA the CPU does not have.

## What changed

Recognizer attention uses `GemmAccumulate`. The AVX2 and AVX-512 kernels
previously streamed the weight matrix once per row. They now pair two rows
so each weight vector is loaded once and applied to two accumulators. Each
output element still walks K in ascending order. Set
`PPOCR_DISABLE_AVX2_GEMM_ACC2` or `PPOCR_DISABLE_AVX512_GEMM_ACC2` to restore
the one-row kernel.

On AArch64 the GEMM used to reload and store C on every K step (four-wide
NEON). It now keeps a 2×16 register tile, which is the same idea as the x86
kernels. `PPOCR_DISABLE_NEON_GEMM_BLOCK` restores the scalar loop.

The recognizer MLP (`ExpandGeluProjectAdd`) was scalar unless the CPU had
AVX-512. AVX2 now runs an 8-wide FMA tile. AArch64 runs the same kernel
16-wide (four NEON vectors) and keeps a 4-wide tail. Unit-stride
convolutions and depthwise layers on AArch64 use a 4-wide interior, and
3×3 stride-2 convolutions use a 4-output NEON tile with stride-2 gathers.
`PPOCR_DISABLE_AVX2_EXPAND_GELU`, `PPOCR_DISABLE_NEON_EXPAND_GELU`,
`PPOCR_DISABLE_NEON_CONV`, `PPOCR_DISABLE_NEON_STRIDE2`, and
`PPOCR_DISABLE_NEON_DW` restore the previous loops.

`tools/bench_cpu.sh` and `ppocr_isa_bench` print the active ISA and time a
32×256×256 GEMM, the matching accumulate, a 960×544 identity RGB normalize,
and optionally a full OCR.

## What was measured

x86 host: 4 vCPU Intel Xeon (family 6, model 207) with AVX2,
AVX-512F/DQ/BW/VL, BMI1, and BMI2. `g++` 13.3. There is no AVX2-only CPU in
that environment. AVX2 is measured by `PPOCR_FORCE_ISA=avx2`, which keeps
`HasAvx512()` false. The AVX2 object file was disassembled and contains no
`zmm` or AVX-512 mask registers (`tests/check_isa_encoding.sh`).

ARM host: 20-core big.LITTLE, 10× Cortex-X925 (up to 3.9 GHz) and 10×
Cortex-A725 (up to 2.8 GHz), `g++` 13.3. `lscpu` flags include NEON
(`asimd`), FP16 (`fphp`, `asimdhp`), dot-product (`asimddp`), `i8mm`,
`bf16`, and `sve`/`sve2`. The SVE vector length is 16 bytes, the same width
as NEON, so the wider path is unrolled NEON rather than SVE. Dot-product and
`i8mm` are integer; this graph is FP32, and an FP16 accumulate would move
the recognized score, so those units are not used. `perf` is not available
to this user (`perf_event_paranoid` is 4) and was left unchanged. Hotspots
below are end-to-end A/B timings with the disable flags.

### GEMM accumulate, 32×256×256, mean of 8 runs (`ppocr_isa_bench`)

| Path | gemm_accumulate |
| --- | --- |
| AVX-512, two-row (new) | 0.070 ms |
| AVX-512, one-row (`PPOCR_DISABLE_AVX512_GEMM_ACC2=1`) | 0.094 ms |
| AVX2, two-row (new) | 0.082 ms |
| AVX2, one-row (`PPOCR_DISABLE_AVX2_GEMM_ACC2=1`) | 0.125 ms |
| scalar (`PPOCR_FORCE_ISA=scalar`) | 0.186 ms |

The same-process `Gemm` timing (not the accumulate kernel) was about 0.044 ms
on AVX-512, 0.059 ms on AVX2, and 0.089 ms on scalar. Identity RGB normalize
of a 960×544 page was about 0.6–0.7 ms on every path; it is not the bottleneck.

### End-to-end PP-OCRv6 tiny, 960×240 “Hello RapidOCR 123”

`ppocr_bench`, `PPOCR_BACKEND=cpu`, 1 warmup + 3 runs, one recognized line:

| ISA | mean |
| --- | --- |
| AVX-512 | 17.3 ms |
| AVX2, before this change (`PPOCR_FORCE_ISA=avx2`) | 136–142 ms |
| AVX2, after the expand-GELU kernel | 31.0 ms |
| scalar | 233 ms |

The spatial 3×3 and 2×2 AVX2 convolution kernels were already vectorized.
A per-op profile of this page showed they were only a few milliseconds.
About 80% of the AVX2 time was `ExpandGeluProjectAdd`, the recognizer MLP
(1×1 expand, GELU, 1×1 project, residual add), which fell through to a
scalar triple loop whenever AVX-512 was off. That kernel is now an 8-wide
AVX2+FMA tile using the same `ErfPs` approximation as `Avx2ExactGelu`, split
across the persistent thread pool. `PPOCR_DISABLE_AVX2_EXPAND_GELU=1`
restores the scalar loop (142 ms, 1 warmup + 3 runs). The new kernel on
the same binary is 31.0 ms mean over 1 warmup + 5 runs (29–34 ms).

Decoded text, score, and box on this image match the AVX-512 and scalar
runs: `Hello RapidOCR 123`, confidence 0.9766, box `23,73,672,86`.

Turning the two-row accumulate off still changes this page by about 1 ms.
The accumulate win shows up on the GEMM shape above. The remaining AVX-512
versus AVX2 gap (17.3 ms versus 31 ms) is the wider AVX-512 tiles on
convolution and this MLP, not an 8× end-to-end claim.

### After d119de0: AVX2 convolution tiles

`perf` is not installed here (`linux-tools` for kernel 6.12.94 is not in
the package index). Hotspots came from `PPOCR_PROFILE=1` and
`PPOCR_PROFILE_E2E=1`.

On the 960×240 line, detector preprocess and DB post-process are each under
1 ms, and recognizer preprocess is about 0.1 ms. Time is the detector
forward pass. The AVX2-versus-AVX-512 gap on that image was a few missing
tiles, not post-process:

- SAME 2×2 convolutions (detector Conv.1 / Conv.2) had an AVX-512 kernel
  and fell through to the generic AVX2 convolution.
- The 5×5 depthwise+pointwise fuse (Conv.78) was AVX-512 only.
- The recognizer vocabulary GEMM already had a packed-B kernel on AVX-512.

On a 1700×2200 page (45 boxes, 18 recognition batches) the balance flips.
Detector preprocess is about 3 ms, DB about 3 ms, detector run about 50–60 ms.
A profiled run puts recognizer wall time at about 180 ms on AVX-512 and
about 275 ms on AVX2 after this change (the profile itself slows the run).
Batches of crops call `Conv2dBatch`. The AVX2 3×3
stride-2 path there used one output channel and a gather, so the folded
batch-norm stem (`fused_conv_batchnorm`, 24→48, and the 3-channel GELU stem)
reread every source row once per filter. The single-image kernel already
had a four-output shuffle tile. An eight-output AVX2 tile was slower on
this page (register pressure) and stays off unless
`PPOCR_ENABLE_AVX2_STRIDE2_TILE8=1`. An eight-filter AVX-512 batch tile
was inside run-to-run noise against the existing four-filter tile, so
AVX-512 batch scheduling is unchanged.

Paired runs of `d119de0` and this tree, same host, `PPOCR_BACKEND=cpu`.
Each cell is the mean of two rounds. Hello is 1 warmup + 5 runs. The page
is 1 warmup + 3 runs.

| Image | ISA | d119de0 | this change |
| --- | --- | --- | --- |
| 960×240 line | AVX-512 | 17.9 ms | 18.0 ms |
| 960×240 line | AVX2 | 29.4 ms | 24.4 ms |
| 1700×2200 page | AVX-512 | 204 ms | 204 ms |
| 1700×2200 page | AVX2 | 377 ms | 293 ms |

Hello mins in those rounds were 16.5–17.4 ms (AVX-512) and 23.0–26.0 ms
(AVX2 after). Text, confidence, and boxes match `d119de0` on both images
and both ISAs, including all 45 page lines. The published 17.3 ms / 31.0 ms
tiny-line figures above are the same host on an earlier run; absolute
times move a few milliseconds between sessions, and the paired table is
the comparison for this change.

Thread sweep on this tree (`PPOCR_BENCH_THREADS`, 4 vCPU). Four threads is
the fastest. The default cap is already `hardware_concurrency`.

| Threads | Hello AVX2 | Hello AVX-512 | Page AVX2 | Page AVX-512 |
| --- | --- | --- | --- | --- |
| 1 | 52 ms | 34 ms | 391 ms | 272 ms |
| 2 | 38 ms | 24 ms | 337 ms | 234 ms |
| 4 | 24 ms | 21 ms | 305 ms | 197 ms |

`PPOCR_DISABLE_AVX2_CONV2X2_SAME`, `PPOCR_DISABLE_AVX2_DWPW5_FUSED`,
`PPOCR_DISABLE_STRIDE2_TILE4`, and `PPOCR_DISABLE_CTC_PACKED_B` restore the
previous kernels. The packed-B flag also disables the AVX-512 vocabulary
pack.

### AArch64 NEON (Cortex-X925 / A725)

Native Release build, `PPOCR_BACKEND=cpu`, PP-OCRv6 tiny, 960×240
“Hello RapidOCR 123”. `ppocr_bench`, 1 warmup + 5 runs:

| Build | mean |
| --- | --- |
| `aae00ca`, NEON (4-wide expand, unit-stride conv, scalar 3×3 stride-2) | 42.6 ms |
| same commit, `PPOCR_FORCE_ISA=scalar` | 138.5 ms |
| this change, NEON | 31.7 ms (29.8–33.5) |
| this change, `PPOCR_DISABLE_NEON_STRIDE2=1` | 39.9–41.6 ms |
| this change, `PPOCR_FORCE_ISA=scalar` | 137.6 ms |

Decoded text, score, and box match the x86 AVX-512, forced-AVX2, and scalar
runs: `Hello RapidOCR 123`, confidence 0.9766, box `23,73,672×86`. The C and
C++ demos, `ctest` (`c_api_smoke`), kernel smoke, and the Python package
(`pytest`, 5 passed, plus a `RapidOCR` call) agree on that text. With the
C++ detection defaults the Python score is 0.9766 and the polygon is
`(23,73)-(695,159)`, which is the same box.

On `aae00ca` the env-flag A/B (1 warmup + 3 runs) put the time here:

| Flag | mean | delta vs 43.2 ms NEON |
| --- | --- | --- |
| `PPOCR_DISABLE_NEON_EXPAND_GELU=1` | 113.2 ms | +70 ms |
| `PPOCR_DISABLE_NEON_CONV=1` | 62.9 ms | +20 ms |
| `PPOCR_DISABLE_NEON_DW=1` | 45.3 ms | +2 ms |
| `PPOCR_DISABLE_NEON_GEMM_BLOCK=1` | 41.1 ms | none |
| `PPOCR_BENCH_THREADS=8` | 58.2 ms | slower |
| `PPOCR_BENCH_THREADS=20` | 52.6 ms | slower |

Expand-GELU is the largest NEON-versus-scalar gap. The 4-wide kernel already
took most of it; widening it to 16 and register-blocking the 1×1 pointwise
loop stay inside the noise of this page (stride-2 off is 40–42 ms, next to
the 42.6 ms baseline). The new 3×3 stride-2 tile is the measured end-to-end
gain, about 10 ms (42.6 ms → 31.7 ms). Default 16 threads is faster than 8
or 20 on this 10+10 layout; the cap was left at 16.

`tools/bench_cpu.sh` on this machine reports `active_isa=neon` by default
and falls through to scalar when `PPOCR_FORCE_ISA=avx2`, which is the
intended dispatch. Microbenchmarks (`gemm_accumulate`, identity RGB) move
around under the host’s other load; the end-to-end numbers above were
repeated and stay in the ranges shown.

#### Earlier qemu-user checks

Before this board was available, the same NEON kernels were timed under
`qemu-aarch64-static`. Those numbers are instruction-execution checks, not
wall-clock for the Cortex-X925 / A725 machine above.

- Standalone old NEON GEMM reload versus the 2×16 register tile, 8×6906×192:
  166 ms → 76 ms, max abs error 0.
- In-tree `Gemm` tile 56–64 ms. Checksum matched the scalar fallback
  (`sum=3.171875`, `max_abs=1.611328`). qemu-user often runs scalar ARM
  faster than NEON, so that scalar 31–34 ms is not a device result.
- Expand-GELU 32×64, plane 1536: 16.3 ms vs scalar 58.2 ms. 3×3 stride-1
  8→8 on 18×20: 1.70 vs 2.09 ms (sum `1008.004807`). 5×5 depthwise: 0.48 vs
  0.82 ms (sum `1007.901022`). Expand-GELU versus `std::erf` max abs
  `2.98e-6`. Full OCR was not run under qemu.
