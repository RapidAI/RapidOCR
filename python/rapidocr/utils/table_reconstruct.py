"""Heuristic table reconstruction from OCR text boxes.

Groups detected text boxes into rows (by Y proximity) and columns (by X
alignment), producing a 2D grid (list of rows, each a list of cell strings).
No neural model required — works with any detector that returns boxes.

Useful as a fallback when PPStructureV3 / SLANet is unavailable.
"""

from __future__ import annotations

from typing import List, Tuple


def reconstruct_table(
    boxes: List,
    texts: List[str],
    scores: List[float] | None = None,
    y_tol: int = 10,
    x_tol: int = 5,
) -> List[List[str]]:
    """Reconstruct a table grid from OCR boxes.

    Args:
        boxes: list of (N,4,2) or (x1,y1,x2,y2) boxes.
        texts: parallel list of recognized strings.
        scores: optional parallel scores (unused, kept for API compat).
        y_tol: vertical tolerance (px) to group boxes into the same row.
        x_tol: horizontal tolerance (px) to align columns.

    Returns:
        List of rows, each row is a list of cell strings (left-to-right).
    """
    if boxes is None or len(boxes) == 0 or not texts:
        return []

    # Normalize boxes to (x1, y1, x2, y2, text)
    items = []
    for b, t in zip(boxes, texts):
        if hasattr(b, "tolist"):
            b = b.tolist()
        if isinstance(b, (list, tuple)) and len(b) >= 4 and isinstance(b[0], (list, tuple)):
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            x1, y1, x2, y2 = b[:4]
        items.append((x1, y1, x2, y2, t))

    # Sort by y then x
    items.sort(key=lambda it: (it[1], it[0]))

    # Group into rows by y proximity
    rows: List[List[Tuple]] = []
    for it in items:
        placed = False
        for row in rows:
            if abs(row[0][1] - it[1]) <= y_tol:
                row.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])

    # Within each row, sort by x and merge adjacent boxes into cells
    grid: List[List[str]] = []
    for row in rows:
        row.sort(key=lambda it: it[0])
        cells: List[str] = []
        cur = row[0][4]
        cur_x2 = row[0][2]
        for it in row[1:]:
            if it[0] - cur_x2 <= x_tol:
                cur = cur + " " + it[4]
                cur_x2 = max(cur_x2, it[2])
            else:
                cells.append(cur.strip())
                cur = it[4]
                cur_x2 = it[2]
        cells.append(cur.strip())
        grid.append(cells)

    return grid


def table_to_text(grid: List[List[str]]) -> str:
    """Render a grid as aligned text (for display)."""
    if not grid:
        return ""
    ncols = max(len(r) for r in grid)
    # pad rows
    padded = [r + [""] * (ncols - len(r)) for r in grid]
    # column widths
    widths = [max(len(padded[r][c]) for r in range(len(padded))) for c in range(ncols)]
    lines = []
    for r in padded:
        cells = [r[c].ljust(widths[c]) for c in range(ncols)]
        lines.append(" | ".join(cells))
    return "\n".join(lines)