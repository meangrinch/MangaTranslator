from collections.abc import Callable

import numpy as np
from PIL import Image

# Section heights are multiples of the width: bubble detection recall peaks around 4-6x,
# dropping for whole strips (small bubbles lost) and page-shaped sections (large ones lost).
SPLIT_RATIO = 8.0
MIN_RATIO, TARGET_RATIO, MAX_RATIO = 3.0, 5.0, 7.0
MIN_GAP_RATIO = 0.045  # exceeds line spacing of bubble-less narration text
BLANK_TOLERANCE = 8


def _pick_gap(free: np.ndarray, min_gap: int, prefer: int) -> int | None:
    """Middle row of the run of free rows (>= min_gap long) nearest to `prefer`."""
    padded = np.concatenate(([False], free, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    middles = [
        int(start + end) // 2
        for start, end in zip(edges[::2], edges[1::2])
        if end - start >= min_gap
    ]
    return min(middles, key=lambda row: abs(row - prefer), default=None)


def find_cut_rows(
    image: Image.Image,
    detect_rows: Callable[[Image.Image], list[tuple[int, int]]] | None = None,
) -> list[int]:
    """Rows at which to split a long strip into sections, or [] if it should stay whole.

    Cuts go through blank gutters when one is in range. Otherwise `detect_rows` (returning
    the (top, bottom) row ranges occupied by text in a crop) picks a row between text.
    """
    w, h = image.size
    if h <= SPLIT_RATIO * w:
        return []

    blank = np.ptp(np.asarray(image.convert("L")), axis=1) <= BLANK_TOLERANCE
    min_gap = max(8, int(w * MIN_GAP_RATIO))
    last = int(h - MIN_RATIO * w)
    cuts: list[int] = []
    pos = 0
    while h - pos > MAX_RATIO * w:
        cut = None
        lo = int(pos + MIN_RATIO * w)
        while cut is None and lo < last:
            hi = min(lo + int((MAX_RATIO - MIN_RATIO) * w), last)
            prefer = int(pos + TARGET_RATIO * w) - lo
            cut = _pick_gap(blank[lo:hi], min_gap, prefer)
            if cut is None and detect_rows is not None:
                free = np.ones(hi - lo, dtype=bool)
                for top, bottom in detect_rows(image.crop((0, lo, w, hi))):
                    free[max(0, int(top)) : max(0, int(bottom))] = False
                cut = _pick_gap(free, min_gap, prefer)
            if cut is not None:
                cut += lo
            lo = hi
        if cut is None:
            # Never cut through text: the rest of the strip stays one section.
            break
        cuts.append(cut)
        pos = cut
    return cuts
