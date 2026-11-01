"""Selection helpers for region swapping.

* :func:`refine_selection` - snaps a rough selection to the real object edges,
  but only inside a narrow band around what the user drew (used live by the
  selection editor after every shape), so it never grows or shrinks far
  beyond the user's intent.
* :func:`extract_object` - the object inside a (possibly loose) loop: a
  colour-model graph cut seeded by the loop, guided by salient-object
  segmentation when the U2-Net "silueta" model is installed. An object cut
  out of a loose loop then does not carry background with it.
* :func:`classify_selection` - is a selection an *object* (cut out and
  placed) or an *area/texture* (sky, grass, a wall: fills the other
  selection's exact shape)?
"""

from __future__ import annotations

import cv2
import numpy as np

from ..imaging.filters import bbox_of, guided_filter


def fill_holes(m: np.ndarray) -> np.ndarray:
    """Binary mask with every region not connected to the border filled."""
    b = (m > 0).astype(np.uint8)
    h, w = b.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = b
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(pad, ff, (0, 0), 1)
    return (b | (pad[1:-1, 1:-1] == 0)).astype(np.uint8)


def _bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _grabcut(rgb: np.ndarray, gc: np.ndarray, iters: int) -> np.ndarray | None:
    fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    if not fg.any() or fg.all():
        return None
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(_bgr(rgb), gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    return ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)


def _resize(img: np.ndarray, s: float, interp=cv2.INTER_AREA) -> np.ndarray:
    if s >= 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


# --------------------------------------------------------------------------- editor snapping
def refine_selection(rgb: np.ndarray, mask: np.ndarray, roi: tuple[int, int, int, int] | None = None, band: float = 12, max_work: int = 640) -> np.ndarray:
    """Snap the selection boundary to image edges inside ``roi`` (x, y, w, h).

    The boundary may move inwards by at most ``band`` pixels and outwards by
    at most ``band / 2`` (a loop drawn *around* an object usually has its
    line outside the object). Thin parts keep their centre line. Returns a
    uint8 0/1 mask of the same size; the input is returned unchanged when
    snapping would change the selection drastically.
    """
    sel = (mask > 0.5).astype(np.uint8)
    H, W = sel.shape
    if sel.sum() < 30 or band < 2:
        return sel
    x, y, w, h = roi if roi else (0, 0, W, H)
    pad = int(band * 2 + 8)
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return sel
    cs = sel[y0:y1, x0:x1]
    if cs.min() == cs.max():
        return sel  # no boundary in this area
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    s = min(1.0, max_work / max(x1 - x0, y1 - y0))
    img_w = _resize(crop, s)
    hw, ww = img_w.shape[:2]
    cs_w = cv2.resize(cs, (ww, hw), interpolation=cv2.INTER_NEAREST) if s < 1 else cs
    bs = max(1.5, band * s)
    din = cv2.distanceTransform(cs_w, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - cs_w, cv2.DIST_L2, 5)
    n, lab = cv2.connectedComponents(cs_w)
    cmax = np.zeros(n, np.float32)
    np.maximum.at(cmax, lab.ravel(), din.ravel())
    core = (cs_w > 0) & (din > np.minimum(bs, 0.5 * cmax[lab]))
    inroi = np.zeros(cs_w.shape, bool)
    rx0, ry0 = int((x - x0) * s), int((y - y0) * s)
    inroi[max(0, ry0) : int((y + h - y0) * s) + 1, max(0, rx0) : int((x + w - x0) * s) + 1] = True
    gc = np.full(cs_w.shape, cv2.GC_PR_BGD, np.uint8)
    gc[cs_w > 0] = cv2.GC_PR_FGD
    gc[core] = cv2.GC_FGD
    gc[dout > bs * 0.5] = cv2.GC_BGD
    gc[~inroi & (cs_w > 0)] = cv2.GC_FGD
    gc[~inroi & (cs_w == 0)] = cv2.GC_BGD
    if not (gc == cv2.GC_BGD).any() and not (gc == cv2.GC_PR_BGD).any():
        return sel
    res = _grabcut(img_w, gc, 3)
    if res is None:
        return sel
    # no new islands; no holes punched into the selection
    n, lab = cv2.connectedComponents(res)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[core | (~inroi & (cs_w > 0))])] = True
    keep[0] = False
    res = keep[lab].astype(np.uint8)
    res = res | (fill_holes(res) & cs_w)
    # back to full resolution with edge-aware upsampling
    if s < 1:
        up = cv2.resize(res.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        up = guided_filter(crop, up, max(2, int(round(1.5 / s))), 1e-3)
        res = (up > 0.5).astype(np.uint8)
    # hard limits around the user's outline (in full-resolution pixels)
    din_f = cv2.distanceTransform(cs, cv2.DIST_L2, 5)
    dout_f = cv2.distanceTransform(1 - cs, cv2.DIST_L2, 5)
    res[(dout_f > band * 0.5 + 1)] = 0
    n, lab = cv2.connectedComponents(cs)
    cmax = np.zeros(n, np.float32)
    np.maximum.at(cmax, lab.ravel(), din_f.ravel())
    res[(cs > 0) & (din_f > np.minimum(band + 1, 0.5 * cmax[lab]))] = 1
    roi_full = np.zeros(cs.shape, bool)
    roi_full[max(0, y - y0) : y + h - y0, max(0, x - x0) : x + w - x0] = True
    res = np.where(roi_full, res, cs).astype(np.uint8)
    a0, a1 = float(cs[roi_full].sum()), float(res[roi_full].sum())
    if a0 > 0 and not (0.5 * a0 <= a1 <= 1.35 * a0 + 1):
        return sel  # would change the selection drastically: trust the user
    out = sel.copy()
    out[y0:y1, x0:x1] = res
    return out


# --------------------------------------------------------------------------- object inside a loop
def extract_object(rgb: np.ndarray, mask: np.ndarray, max_work: int = 560, saliency: bool = True) -> tuple[np.ndarray, dict]:
    """The object inside a selection (uint8 0/1 mask) and a small report.

    Everything inside the selection starts as "probably object", everything
    outside as background (with a thin margin in case the line cut into the
    object); the graph cut then removes the background that the loop
    enclosed. Falls back to the selection itself when no object separates
    clearly from its surroundings.
    """
    sel = (mask > 0.5).astype(np.uint8)
    area = int(sel.sum())
    if area < 64:
        return sel, {"method": "selection"}
    side = float(np.sqrt(area))
    pad = int(max(10, 0.3 * side))
    y0, y1, x0, x1 = bbox_of(sel, pad=pad)
    crop = rgb[y0:y1, x0:x1].astype(np.float32)
    cs = sel[y0:y1, x0:x1]
    s = min(1.0, max_work / max(crop.shape[:2]))
    img_w = _resize(crop, s)
    hw, ww = img_w.shape[:2]
    cs_w = cv2.resize(cs, (ww, hw), interpolation=cv2.INTER_NEAREST) if s < 1 else cs
    dout = cv2.distanceTransform(1 - cs_w, cv2.DIST_L2, 5)
    din = cv2.distanceTransform(cs_w, cv2.DIST_L2, 5)
    margin = max(2.0, 0.025 * side * s)
    gc = np.full(cs_w.shape, cv2.GC_BGD, np.uint8)
    gc[dout <= margin] = cv2.GC_PR_BGD
    gc[cs_w > 0] = cv2.GC_PR_FGD
    info: dict = {"method": "graphcut"}
    sal = None
    if saliency:
        try:
            from ..models import manager

            if manager.is_installed("segment-silueta"):
                from ..imaging.u2net import predict

                sal = predict(img_w, "segment-silueta")
        except Exception:  # noqa: BLE001 - saliency is only a hint
            sal = None
    if sal is not None:
        inside = cs_w > 0
        sal_in = float(sal[inside].mean())
        strong = inside & (sal > 0.6)
        if strong.sum() > 0.08 * inside.sum() and sal_in > 0.12:
            gc[inside & (sal < 0.12)] = cv2.GC_PR_BGD
            gc[strong & (din > max(2.0, 0.15 * din.max()))] = cv2.GC_FGD
            info["method"] = "graphcut+saliency"
    res = _grabcut(img_w, gc, 5)
    if res is None:
        return sel, {"method": "selection", "reason": "graph cut failed"}
    allowed = cs_w | (dout <= margin).astype(np.uint8)
    res &= allowed
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # reconnect parts split by a thin occluder in front (a hoop rim, a net string, a pole)
    rc = max(1, int(0.02 * side * s))
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rc + 1, 2 * rc + 1))) & allowed
    n, lab, stats, _ = cv2.connectedComponentsWithStats(res)
    if n < 2:
        return sel, {"method": "selection", "reason": "nothing separated"}
    # every substantial part inside the loop (not only the largest one)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= max(0.03 * float(cs_w.sum()), 0.05 * float(areas.max()))
    res = fill_holes(keep[lab].astype(np.uint8))
    frac = float(res.sum()) / max(float(cs_w.sum()), 1.0)
    info["fill"] = round(frac, 3)
    if frac < 0.12:
        return sel, {**info, "method": "selection", "reason": "object too small for the selection"}
    if s < 1:
        up = cv2.resize(res.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        up = guided_filter(crop, up, max(2, int(round(1.5 / s))), 1e-3)
        res = (up > 0.5).astype(np.uint8)
    out = np.zeros_like(sel)
    out[y0:y1, x0:x1] = res
    return out, info


# --------------------------------------------------------------------------- object or area?
def _lab_hist(lab: np.ndarray, w: np.ndarray) -> np.ndarray:
    m = (w > 0).astype(np.uint8)
    h = cv2.calcHist([lab], [0, 1, 2], m, [6, 8, 8], [0, 256, 0, 256, 0, 256])
    return h / max(float(h.sum()), 1.0)


def classify_selection(rgb: np.ndarray, mask: np.ndarray) -> tuple[str, dict]:
    """Decide whether a selection is an ``"object"`` or an ``"area"`` (texture/background).

    Large selections and selections running along the image border (sky,
    ground, walls) are areas; so are selections whose inside looks like their
    surroundings (the texture continues across the line). Everything else is
    an object.
    """
    sel = (mask > 0.5).astype(np.uint8)
    H, W = sel.shape
    frac = float(sel.mean())
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner = cv2.erode(np.pad(sel, 1), k)[1:-1, 1:-1]
    boundary = sel & (1 - inner)
    frame = np.zeros_like(sel)
    frame[[0, -1], :] = 1
    frame[:, [0, -1]] = 1
    border_frac = float((boundary & frame).sum()) / max(float(boundary.sum()), 1.0)
    info = {"area_fraction": round(frac, 3), "border_fraction": round(border_frac, 3)}
    if frac > 0.4 or border_frac > 0.3:
        return "area", {**info, "reason": "large or along the image border"}
    side = float(np.sqrt(max(sel.sum(), 1)))
    s = min(1.0, 700 / max(H, W))
    small = _resize(rgb.astype(np.float32), s)
    sm = cv2.resize(sel, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) if s < 1 else sel
    r = max(3, int(0.25 * side * s))
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ring_out = cv2.dilate(sm, kk) & (1 - sm)
    ring_in = sm & (1 - cv2.erode(sm, kk))
    if ring_out.sum() < 30 or ring_in.sum() < 30:
        return "object", info
    lab = cv2.cvtColor((np.clip(small, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    d = float(cv2.compareHist(_lab_hist(lab, ring_in), _lab_hist(lab, ring_out), cv2.HISTCMP_BHATTACHARYYA))
    info["inside_outside_difference"] = round(d, 3)
    if d < 0.12:  # neighbouring objects of the same kind (flowers in a field) are still objects
        return "area", {**info, "reason": "texture continues across the outline"}
    return "object", info
