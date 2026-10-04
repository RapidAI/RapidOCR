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
AVX-512. AVX2 now runs an 8-wide FMA tile, and AArch64 runs a 4-wide NEON
tile. Unit-stride convolutions and depthwise layers on AArch64 use a 4-wide
interior as well. `PPOCR_DISABLE_AVX2_EXPAND_GELU`,
`PPOCR_DISABLE_NEON_EXPAND_GELU`, `PPOCR_DISABLE_NEON_CONV`, and
`PPOCR_DISABLE_NEON_DW` restore the previous loops.

`tools/bench_cpu.sh` and `ppocr_isa_bench` print the active ISA and time a
32×256×256 GEMM, the matching accumulate, a 960×544 identity RGB normalize,
and optionally a full OCR.

## What was measured

Host: 4 vCPU Intel Xeon (family 6, model 207) with AVX2, AVX-512F/DQ/BW/VL,
BMI1, and BMI2. `g++` 13.3. There is no ARM board and no AVX2-only CPU in
this environment. AVX2 is measured by `PPOCR_FORCE_ISA=avx2`, which keeps
`HasAvx512()` false. The AVX2 object file was disassembled and contains no
`zmm` or AVX-512 mask registers (`tests/check_isa_encoding.sh`).

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

### AArch64 NEON

No ARM CPU was available. `src/kernels.cpp` cross-compiles with
`aarch64-linux-gnu-g++ -O3` and the GEMM object contains `fmla`. The same
8×6906×192 recognizer-vocabulary shape was timed under `qemu-aarch64-static`:

- Standalone copy of the old NEON reload loop versus the new register tile:
  166 ms → 76 ms. That is the algorithm before/after. The tile result matched
  the reload loop (max abs error 0).
- In-tree `Gemm` linked against that object: register tile 56–64 ms. Checksum
  and max abs matched the scalar fallback
  (`PPOCR_DISABLE_NEON_GEMM_BLOCK=1`, 31–34 ms) exactly (`sum=3.171875`,
  `max_abs=1.611328`). qemu-user's translator often runs scalar ARM faster
  than NEON, so the scalar number is not a wall-clock result for a Cortex or
  Neoverse core.

The same cross-compile covers the convolution and MLP paths that were scalar
on AArch64. `NeonConv2d` and `NeonDepthwiseConv` are 4-wide FMA interiors for
unit-stride 2/3/5/7 convolutions and 3/5/7/9 depthwise layers.
`NeonExpandGeluProjectAdd` is the 4-wide form of the MLP kernel above, with
the same erf polynomial as AVX2. `PPOCR_DISABLE_NEON_CONV`,
`PPOCR_DISABLE_NEON_DW`, and `PPOCR_DISABLE_NEON_EXPAND_GELU` restore the
scalar loops.

Under `qemu-aarch64-static` (not cycle-accurate), in-tree kernels versus
those scalar fallbacks:

| Kernel | NEON | scalar fallback |
| --- | --- | --- |
| Expand-GELU 32×64, plane 1536 | 16.3 ms | 58.2 ms |
| 3×3 stride-1 conv, 8→8, 18×20, pad 1 | 1.70 ms | 2.09 ms |
| 5×5 depthwise, 8 channels, 18×20, pad 2 | 0.48 ms | 0.82 ms |

Convolution and depthwise sums matched the scalar fallback exactly
(`1008.004807` and `1007.901022`). Expand-GELU versus a `std::erf` reference
had max abs error `2.98e-6` (the scalar fallback itself was `1.19e-7` under
`-ffast-math`). Full OCR was not run under qemu.

qemu-user is not cycle-accurate. It does execute the NEON instructions. The
GEMM NEON-versus-NEON ratio matches the drop in C-matrix traffic. The small
convolution shapes above are too short for qemu to show a large ratio; the
sums are the correctness check.
