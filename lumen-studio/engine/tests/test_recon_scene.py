"""3D reconstruction orchestration (recon/scene.py) and the novel-view evaluation harness (tools/recon_eval)."""

import sys
from pathlib import Path

import cv2
import numpy as np

from conftest import load, needs_depth, needs_testdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # engine/ (for tools.recon_eval)


def _person_scene():
    """Wall at disparity 0.15, a floor plane receding towards the horizon, a person (disparity 0.7)
    standing on it; only the head is 'segmented'."""
    h, w = 240, 320
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    d = np.full((h, w), 0.15, np.float32)
    floor = yy > 150
    d[floor] = 0.15 + (yy[floor] - 150) / (h - 150) * 0.75  # floor gets nearer towards the bottom
    person = np.zeros((h, w), bool)
    cv2.ellipse(person.view(np.uint8), (160, 60), (18, 22), 0, 0, 360, 1, -1)  # head
    cv2.rectangle(person.view(np.uint8), (135, 80), (185, 200), 1, -1)  # body down to the floor
    d[person] = 0.7
    head = np.zeros((h, w), bool)
    cv2.ellipse(head.view(np.uint8), (160, 60), (18, 22), 0, 0, 360, 1, -1)
    rgb = np.full((h, w, 3), 0.5, np.float32)
    return rgb, d, head, person, floor


def test_complete_foreground_grows_partial_person_but_not_floor():
    from lumen_engine.recon.scene import _complete_foreground

    rgb, d, head, person, floor = _person_scene()
    sky = np.zeros_like(head)
    fg, people, grown = _complete_foreground(rgb, d, head.copy(), head.copy(), sky, 0.045)
    body = person & ~head
    assert grown > 0.02
    assert (fg & body).sum() > 0.9 * body.sum()  # the unsegmented body joined the head
    assert (people & body).sum() > 0.9 * body.sum()  # ... as part of the person
    assert (fg & floor & ~person).sum() < 0.02 * (floor & ~person).sum()  # the floor it stands on did not
    assert not (fg & ~person & ~floor).any() or (fg & ~person & ~floor).sum() < 50  # nor the wall
    # nothing segmented -> nothing grown; a separate object at another depth is not merged
    fg0, _, g0 = _complete_foreground(rgb, d, np.zeros_like(head), np.zeros_like(head), sky, 0.045)
    assert g0 == 0.0 and not fg0.any()
    d2 = d.copy()
    d2[person & ~head] = 0.45  # body much farther than the head: a different surface
    fg2, _, _ = _complete_foreground(rgb, d2, head.copy(), head.copy(), sky, 0.045)
    assert (fg2 & body).sum() < 0.1 * body.sum()


def test_harness_metrics_on_synthetic_scene():
    """The harness reads the CPU renderer's G-buffer (render._frame); this guards that contract."""
    from lumen_engine.recon import render as R
    from tools.recon_eval.cameras import Cam, standard_cameras
    from tools.recon_eval.harness import render_view, stretch_flags
    from tools.recon_eval.metrics import SceneGeometry, overlay, view_metrics

    sc, tm, tb, tk = R._synthetic_scene()
    geo = SceneGeometry(sc, tm.rgb, tb.rgb)
    cm = geo.container_metrics()
    assert 0.0 <= cm["behind_fraction"] <= 1.0
    W, H = 96, 64
    img, gb = render_view(R, sc, tm, tb, tk, Cam("ref"), W, H, 60.0, 8.0, margin_fade=0.0)
    assert gb is not None and gb["layer"].size == W * H, "render._frame no longer exposes glayer/gu/gv"
    m, masks = view_metrics(img, gb["layer"], gb["u"], gb["v"], geo, *stretch_flags(R, sc, Cam("ref"), W, H, 60.0, 8.0))
    assert m["empty"] < 0.01 and m["stretched"] == 0.0
    moved = Cam("x", (0.3, 0.0, 0.0), yaw=30.0)
    img2, gb2 = render_view(R, sc, tm, tb, tk, moved, W, H, 60.0, 8.0)
    m2, masks2 = view_metrics(img2, gb2["layer"], gb2["u"], gb2["v"], geo, *stretch_flags(R, sc, moved, W, H, 60.0, 8.0))
    assert m2["guessed"] > m["guessed"]
    assert overlay(img2, masks2).shape == img2.shape
    cams = standard_cameras(2.5)
    assert len(cams) == 25
    assert {"orbitL35", "orbitR35", "orbitL50", "orbitR50"} <= {c.name for c in cams}


@needs_depth
@needs_testdata
def test_reconstruct_reports_stage_timings_and_container():
    from lumen_engine.recon import render as R
    from lumen_engine.recon.scene import ReconSettings, reconstruct

    rgb = cv2.resize(load("scenes/home.jpg"), (320, 240), interpolation=cv2.INTER_AREA)
    out = reconstruct(rgb, ReconSettings(quality="fast", detail_fusion=False))
    rep = out["report"]
    assert rep["time_total"] > 0 and "Estimating depth" in rep["timings"]
    assert sum(rep["timings"].values()) <= rep["time_total"] + 0.5
    sc = R.parse_scene(out["scene"])  # the container parses with the CPU renderer
    assert sc.disp.shape == (out["header"]["grid_h"], out["header"]["grid_w"])
    assert np.isfinite(sc.disp).all() and sc.keep.any()
