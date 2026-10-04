#!/bin/sh
# Time each legal runtime path. Extra arguments are forwarded to ppocr_isa_bench
# (DET REC DICT IMAGE RUNS) when an end-to-end OCR number is wanted.
set -eu
bench=$1
shift
echo "===== default (best legal ISA) ====="
"$bench" "$@"
echo "===== PPOCR_FORCE_ISA=avx2 ====="
PPOCR_FORCE_ISA=avx2 "$bench" "$@"
echo "===== PPOCR_FORCE_ISA=avx2 PPOCR_DISABLE_AVX2_GEMM_ACC2=1 ====="
PPOCR_FORCE_ISA=avx2 PPOCR_DISABLE_AVX2_GEMM_ACC2=1 "$bench" "$@"
echo "===== PPOCR_FORCE_ISA=scalar ====="
PPOCR_FORCE_ISA=scalar "$bench" "$@"
echo "===== PPOCR_DISABLE_AVX512_GEMM_ACC2=1 ====="
PPOCR_DISABLE_AVX512_GEMM_ACC2=1 "$bench" "$@"
