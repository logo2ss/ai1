import cv2
import numpy as np
import pytest

from lumen_engine.face.fallback import template
from lumen_engine.face.landmarks import Face, topology
from lumen_engine.face.settings import FaceSwapSettings
from lumen_engine.face.swap import swap_face
from lumen_engine.face.geometry import apply_affine


def fixture_face():
    pts = template()['template478'].copy()
    xy=pts[:,:2]; xy -= (xy.min(0)+xy.max(0))/2
    pts[:, :2] = xy * (140 / np.ptp(xy,axis=0).max()) + (160, 150)
    return Face(pts, np.eye(4))


def test_pure_preserves_all_aligned_pixels_and_outside(monkeypatch):
    from lumen_engine.face import parsing
    monkeypatch.setattr(parsing, 'available', lambda: False)
    src_face = fixture_face()
    # Non-trivial similarity alignment and patterned source: blur/tone changes fail.
    M = cv2.getRotationMatrix2D((160,160), 6, 0.85); M[:,2] += (9,4)
    pts = src_face.points.copy(); pts[:,:2] = apply_affine(M, pts[:,:2])
    dst_face = Face(pts, np.eye(4))
    yy,xx=np.mgrid[:320,:320]
    src=np.stack([0.4+0.1*np.sin(xx*0.2), 0.4+0.1*np.sin(yy*0.3), np.full_like(xx,0.3,dtype=float)],-1).astype(np.float32)
    dst=np.full_like(src,(0.62,0.43,0.29))
    out,rep,dbg=swap_face(src,src_face,dst,dst_face,FaceSwapSettings(mode='pure',force=True))
    aligned=np.clip(cv2.warpAffine(src,dbg['similarity'],(320,320),flags=cv2.INTER_CUBIC),0,1)
    protected=dbg['mask']>0.5
    assert protected.sum()>1000
    assert np.max(np.abs(out[protected]-aligned[protected])) < 1e-5
    assert rep['pure']['max_face_change']==0
    assert np.array_equal(out[:5],dst[:5])
    assert np.max(np.abs(out[~protected]-dst[~protected]))>0.001


def test_rejects_unknown_mode():
    from lumen_engine.errors import InvalidInputError
    with pytest.raises(InvalidInputError): FaceSwapSettings.from_dict({'mode':'other'})
