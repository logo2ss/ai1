import cv2
import numpy as np
from lumen_engine.region.swap import insert_region, RegionSwapSettings


def test_area_does_not_borrow_pixels_outside_source_selection():
    # White source rectangle on a red background; cover fit of an irregular
    # selection used to leak red from outside the user's selected region.
    src=np.full((120,120,3),(1,0,0),np.float32)
    mask=np.zeros((120,120),np.float32);cv2.circle(mask,(60,60),35,1,-1)
    src[mask>0]=(0.6,0.6,0.6)
    dst=np.full_like(src,0.25);dm=np.zeros_like(mask);dm[25:95,25:95]=1
    s=RegionSwapSettings(mode='area',snap_edges=False,edge_softness=0,color_harmony=0,lighting_match=0,blend_mode='matte',match_grain=False,match_sharpness=False,light_wrap=0)
    out,rep,dbg=insert_region(src,mask,dst,dm,s)
    # Corners outside the warped selection are not covered, with an explicit warning.
    assert dbg['alpha'][30,30] < 0.05
    assert any('does not fully cover' in w for w in rep['warnings'])
    assert np.array_equal(out[:10],dst[:10])
    assert out[60,60].mean() > 0.4
