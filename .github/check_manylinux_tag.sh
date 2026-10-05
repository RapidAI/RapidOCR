#!/bin/bash
# Fail unless every wheel in the directory is a manylinux2014 / manylinux_2_17 wheel.
set -euo pipefail

dir=${1:-wheelhouse}
python3 -m pip install -q auditwheel
mapfile -t wheels < <(find "$dir" -type f -name '*.whl' | sort)
if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "no wheels under $dir" >&2
  exit 1
fi
for wheel in "${wheels[@]}"; do
  base=$(basename "$wheel")
  if ! echo "$base" | grep -Eq 'manylinux2014|manylinux_2_17'; then
    echo "platform tag missing from filename: $base" >&2
    exit 1
  fi
  show=$(python3 -m auditwheel show "$wheel")
  printf '%s\n' "$show"
  if ! printf '%s\n' "$show" | grep -Eq 'manylinux2014|manylinux_2_17'; then
    echo "auditwheel show did not report manylinux2014/manylinux_2_17: $base" >&2
    exit 1
  fi
  echo "tag_ok $base"
done
