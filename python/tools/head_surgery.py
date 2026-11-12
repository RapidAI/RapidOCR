"""Head surgery: prune the recognition (Rec) CTC projection vocabulary.

Given a source character dictionary (the baseline model's vocab) and a target
reduced dictionary (e.g. Spanish-only), this module computes the index mapping
between them and slices the final linear projection weight/bias matrices so the
resulting model only emits target characters.

The actual ONNX graph rewriting is intentionally left to the training pipeline
(running on GPU servers); this module provides the reusable, testable core:
index mapping + numpy weight slicing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def read_char_list(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n").rstrip("\r") for line in f if line.strip()]


def build_index_mapping(
    source_chars: List[str], target_chars: List[str]
) -> Dict[str, int]:
    """Map each target char to its index in the source char list.

    Returns a dict {target_char: source_index}. Characters not found in source
    are reported (returned separately) by the caller.
    """
    src_index = {c: i for i, c in enumerate(source_chars)}
    mapping: Dict[str, int] = {}
    missing: List[str] = []
    for tc in target_chars:
        if tc in src_index:
            mapping[tc] = src_index[tc]
        else:
            missing.append(tc)
    if missing:
        raise ValueError(f"Target chars not found in source dict: {missing}")
    return mapping


def compute_target_indices(
    source_chars: List[str], target_chars: List[str]
) -> List[int]:
    """Return source indices (in target order) for the target vocabulary."""
    mapping = build_index_mapping(source_chars, target_chars)
    return [mapping[c] for c in target_chars]


def slice_linear_weights(
    weight: np.ndarray, bias: np.ndarray, source_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice a (out_features, in_features) weight and (out_features,) bias.

    `source_indices` selects which output rows (characters) to keep, in the
    target order. Blank/CTC handling is the caller's responsibility.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be 2D (out_features, in_features)")
    if bias.ndim != 1:
        raise ValueError("bias must be 1D (out_features,)")
    if weight.shape[0] != bias.shape[0]:
        raise ValueError("weight/bias out_features mismatch")
    if any(i < 0 or i >= weight.shape[0] for i in source_indices):
        raise ValueError("source_indices out of range")
    return weight[source_indices, :].copy(), bias[source_indices].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute head-surgery index mapping.")
    parser.add_argument("--source-dict", type=Path, required=True)
    parser.add_argument("--target-dict", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    src = read_char_list(args.source_dict)
    tgt = read_char_list(args.target_dict)
    indices = compute_target_indices(src, tgt)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"source": src, "target": tgt, "source_indices": indices}, f, ensure_ascii=False, indent=2)
    print(f"Wrote mapping for {len(tgt)} target chars -> {args.out}")


if __name__ == "__main__":
    main()