# -*- encoding: utf-8 -*-
"""Tests for heuristic table reconstruction."""
import sys
from pathlib import Path


root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from rapidocr.utils.table_reconstruct import reconstruct_table, table_to_text  # noqa: E402


def test_reconstruct_simple():
    boxes = [(0, 0, 10, 10), (20, 0, 30, 10), (0, 20, 10, 30), (20, 20, 30, 30)]
    texts = ["A", "B", "C", "D"]
    grid = reconstruct_table(boxes, texts)
    assert grid == [["A", "B"], ["C", "D"]]


def test_reconstruct_single_row():
    boxes = [(0, 0, 10, 10), (20, 0, 30, 10)]
    texts = ["X", "Y"]
    grid = reconstruct_table(boxes, texts)
    assert grid == [["X", "Y"]]


def test_empty():
    assert reconstruct_table([], []) == []


def test_table_to_text():
    grid = [["A", "B"], ["C", "D"]]
    out = table_to_text(grid)
    assert "A" in out and "D" in out
    assert "|" in out
