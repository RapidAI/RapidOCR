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
| AVX2 (`PPOCR_FORCE_ISA=avx2`) | 136 ms |
| scalar | 233 ms |

Turning the two-row accumulate off changed this page by about 1 ms, inside
the run-to-run spread. The page is convolution-bound. The accumulate win
shows up on the GEMM shape above, which is the attention/GEMM kernel, not on
this one short line. The large AVX-512 versus AVX2 gap is the existing
AVX-512 convolution kernels, selected at runtime; it is not a claim that the
two-row accumulate produced an 8× end-to-end speedup.

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

qemu-user is not cycle-accurate. It does execute the NEON instructions, and
the NEON-versus-NEON ratio matches the drop in C-matrix traffic. Full OCR was
not run under qemu.
