"""Labelled contact sheets (plain OpenCV drawing, RGB uint8 in / JPEG out)."""

from __future__ import annotations

import cv2
import numpy as np

BG = (24, 24, 28)


def fit(img: np.ndarray, w: int, h: int | None = None) -> np.ndarray:
    ih, iw = img.shape[:2]
    if h is None:
        h = int(round(w * ih / iw))
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.full((h, w, 3), BG, np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y : y + nh, x : x + nw] = r
    return out


def label(tile: np.ndarray, lines: list[str], scale: float = 0.42, top: bool = True) -> np.ndarray:
    t = tile.copy()
    lh = int(16 * scale / 0.42)
    hbar = lh * len(lines) + 4
    y0 = 0 if top else t.shape[0] - hbar
    t[y0 : y0 + hbar] = (t[y0 : y0 + hbar].astype(np.float32) * 0.35).astype(np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(t, s, (4, y0 + lh * (i + 1) - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def text_panel(lines: list[str], w: int, h: int, scale: float = 0.42) -> np.ndarray:
    p = np.full((h, w, 3), BG, np.uint8)
    lh = int(16 * scale / 0.42)
    for i, s in enumerate(lines):
        y = lh * (i + 1)
        if y > h - 2:
            break
        cv2.putText(p, s, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), 1, cv2.LINE_AA)
    return p


def grid(tiles: list[np.ndarray], cols: int, pad: int = 4) -> np.ndarray:
    if not tiles:
        return np.full((10, 10, 3), BG, np.uint8)
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row = []
        for t in tiles[i : i + cols]:
            c = np.full((th, tw, 3), BG, np.uint8)
            c[: t.shape[0], : t.shape[1]] = t
            row.append(c)
        while len(row) < cols:
            row.append(np.full((th, tw, 3), BG, np.uint8))
        rows.append(np.hstack([np.pad(c, ((pad, pad), (pad, pad), (0, 0)), constant_values=BG[0]) for c in row]))
    return np.vstack(rows)


def hrow(tiles: list[np.ndarray], pad: int = 4) -> np.ndarray:
    """Tiles of different widths side by side (top-aligned)."""
    h = max(t.shape[0] for t in tiles)
    return np.hstack([np.pad(t, ((pad, pad + h - t.shape[0]), (pad, pad), (0, 0)), constant_values=BG[0]) for t in tiles])


def stack(parts: list[np.ndarray]) -> np.ndarray:
    w = max(p.shape[1] for p in parts)
    return np.vstack([np.pad(p, ((0, 0), (0, w - p.shape[1]), (0, 0)), constant_values=BG[0]) for p in parts])


def save_jpg(path, rgb: np.ndarray, q: int = 86) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])


def load_rgb(path) -> np.ndarray | None:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
