#!/bin/sh
# Fail if the baseline or AVX2 object files contain AVX-512 encodings.
# AVX-512 lives only in kernels_avx512.cpp, which is entered after CPUID.
set -eu
baseline=$1
avx2=$2
avx512=$3

fail() {
  echo "ISA encoding check failed: $1" >&2
  exit 1
}

if ! command -v objdump >/dev/null 2>&1; then
  echo "objdump is required" >&2
  exit 1
fi

# zmm registers and AVX-512 mask registers are the illegal encodings for an
# AVX2-only machine. xmm/ymm are legal in the AVX2 object.
if objdump -d "$baseline" | grep -E -q '%zmm|%k[0-7]'; then
  fail "baseline kernels.cpp contains AVX-512 opcodes"
fi
if objdump -d "$avx2" | grep -E -q '%zmm|%k[0-7]'; then
  fail "kernels_avx2.cpp contains AVX-512 opcodes"
fi
if objdump -d "$baseline" | grep -E -q '%ymm'; then
  fail "baseline kernels.cpp contains AVX/ymm opcodes"
fi
if ! objdump -d "$avx512" | grep -E -q '%zmm'; then
  fail "kernels_avx512.cpp did not emit AVX-512"
fi
echo "isa encoding check passed"
