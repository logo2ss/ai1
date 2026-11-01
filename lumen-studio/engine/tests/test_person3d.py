import math
import cv2
import numpy as np
import pytest

from lumen_engine.recon.person3d import build_person3d, Person3DSettings
from lumen_engine.recon.person_render import load_person, render_person


def test_closed_person_volume_and_oblique_views(monkeypatch):
    from lumen_engine.models import manager
    monkeypatch.setattr(manager, 'is_installed', lambda _: False)
    mask = np.zeros((180, 120), np.float32)
    cv2.ellipse(mask, (60, 50), (22, 38), 0, 0, 360, 1, -1)
    cv2.rectangle(mask, (27, 60), (92, 165), 1, -1)
    cv2.rectangle(mask, (53, 130), (65, 179), 0, -1)
    rgb = np.full((180, 120, 3), (0.1, 0.8, 0.15), np.float32)
    rgb[mask > 0] = (0.7, 0.35, 0.24)
    out = build_person3d(rgb, mask, Person3DSettings(detail=100))
    v, uv, f = model = load_person(out['person'])
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    assert np.ptp(v[:, 2]) > 0.1
    tex = cv2.cvtColor(cv2.imdecode(np.frombuffer(out['texture'], np.uint8), 1), cv2.COLOR_BGR2RGB)
    for degrees in (-50, 0, 50):
        a = math.radians(degrees)
        frame, _ = render_person(model, tex, (2.3*math.sin(a), 0, 2.3*math.cos(a)), width=320, height=360)
        person = np.any(frame != (14, 17, 23), axis=2)
        assert person.sum() > 8000
        assert np.isfinite(frame).all()
        assert frame[person, 0].mean() > frame[person, 1].mean()


def test_person_empty_and_invalid_settings():
    from lumen_engine.errors import InvalidInputError
    with pytest.raises(InvalidInputError):
        build_person3d(np.zeros((32, 32, 3), np.float32), np.zeros((32, 32), np.float32))
    with pytest.raises(InvalidInputError):
        Person3DSettings.from_dict({'thickness': float('nan')})
