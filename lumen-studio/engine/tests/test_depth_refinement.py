"""Accuracy regressions: relative-depth fusion, thin geometry and sky semantics."""
import numpy as np
from lumen_engine.depth.estimator import _align_relative_depth
from lumen_engine.recon import geometry as G


def test_depth_alignment_preserves_order_and_rejects_reversal():
    yy, xx = np.mgrid[:80,:120].astype(np.float32)
    truth = .3+.002*xx+.001*yy
    good, confidence = _align_relative_depth((truth-8)/3, truth)
    assert confidence > .95
    assert np.abs(good-truth).max() < 1e-5
    bad, confidence = _align_relative_depth(-truth, truth)
    assert confidence == 0 and np.array_equal(bad,truth)
    flat, confidence = _align_relative_depth(np.ones_like(truth),truth)
    assert confidence == 0 and np.array_equal(flat,truth)


def test_depth_alignment_resists_local_occluder_outliers():
    yy, xx = np.mgrid[:80,:120].astype(np.float32)
    truth = .3+.002*xx+.001*yy
    pred = truth*2+4
    pred[20:35,30:60] += 1.5
    out, confidence = _align_relative_depth(pred,truth)
    valid = np.ones(truth.shape,bool);valid[20:35,30:60]=False
    assert confidence > 0
    assert np.abs(out-truth)[valid].mean() < 1e-4


def test_thin_foreground_cannot_take_another_objects_depth():
    d=np.full((100,160),.1,np.float32)
    fg=np.zeros(d.shape,bool)
    fg[15:85,15:65]=True;d[fg]=.85
    fg[20:80,125:128]=True;d[20:80,125:128]=.35
    out=G._snap_inside(d,fg,5)
    assert np.array_equal(out[20:80,125:128],d[20:80,125:128])
    assert np.array_equal(out[~fg],d[~fg])


def test_planar_regularization_never_changes_protected_subject(monkeypatch):
    rgb=np.full((80,120,3),.5,np.float32)
    d=np.full((80,120),.2,np.float32);d[20:60,40:80]=.7
    allowed=np.ones(d.shape,bool);allowed[20:60,40:80]=False
    # Even an over-aggressive plane estimate cannot flatten the protected subject.
    monkeypatch.setattr(G,'regularize_planes',lambda rgb,d,**kw:np.full_like(d,.2))
    out=G._sharpen_depth_edges(rgb,d,.045,mask=allowed)
    assert np.array_equal(out[~allowed],d[~allowed])


def test_sky_reaches_border_but_white_room_does_not_become_sky():
    rgb=np.full((80,120,3),(.9,.9,.9),np.float32)
    d=np.full((80,120),.1,np.float32);d[:30]=0
    assert not G.detect_sky(rgb,d,hint=np.zeros_like(d)).any()
    rgb[:30]=(.5,.7,1)
    hint=np.zeros_like(d);hint[:30]=1
    sky=G.detect_sky(rgb,d,hint=hint)
    assert sky[0].mean() > .95


def test_cpu_preview_preserves_thin_silhouette_quads():
    from lumen_engine.recon.render import _preview_detail_keep
    # One narrow bar whose fine cells lie inside otherwise cut coarse tiles.
    fine=np.zeros((12,24),np.uint8);fine[:,10:12]=1
    rows=np.arange(0,13,3);cols=np.arange(0,25,3)
    coarse=np.zeros((4,8),np.uint8)
    detail=_preview_detail_keep(fine,coarse,rows,cols)
    assert np.array_equal(detail,fine)
    full=np.ones_like(fine);coarse=np.ones_like(coarse)
    assert not _preview_detail_keep(full,coarse,rows,cols).any()


def test_large_enclosed_reflection_pocket_keeps_floor_and_foreground():
    from lumen_engine.recon.geometry import Plane, flatten_planes
    h,w=120,180
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    truth=.25+.5*yy/(h-1)
    reflection=np.zeros((h,w),bool);reflection[65:83,65:95]=True
    foreground=np.zeros_like(reflection);foreground[35:55,25:45]=True
    d=truth.copy();d[reflection]-=.2;d[foreground]+=.2
    region=~reflection & ~foreground
    p=Plane(0,.25,.5,kind='floor',support=float(region.mean()),mask=region)
    out=flatten_planes(d,[p])
    assert np.abs(out[reflection]-truth[reflection]).mean() < .005
    assert np.array_equal(out[foreground],d[foreground])
    # The same depth hole in a wall may be a real opening, never fill it.
    wall=Plane(0,.25,.5,kind='wall',support=float(region.mean()),mask=region)
    out=flatten_planes(d,[wall])
    assert np.array_equal(out[reflection],d[reflection])


def test_preview_and_full_frames_keep_the_same_narrow_foreground():
    from lumen_engine.recon import render as R
    h,w=120,800
    disp=np.full((h,w),.2,np.float32);disp[:,401:404]=.8
    kh=np.abs(np.diff(disp,axis=1))>.1
    keep=~(kh[:-1]|kh[1:])
    header={'aspect':w/h,'extent_x':1.,'extent_y':1.}
    sc=R.SceneData(header,disp,keep.astype(np.uint8),np.full_like(disp,.2),np.ones_like(keep,dtype=np.uint8),None)
    image=np.zeros((h,w,3),np.uint8);image[:]=(20,150,20);image[:,401:404]=(240,20,20)
    tex=R.MipTexture(image);bg=R.MipTexture(np.full((h,w,3),(20,150,20),np.uint8))
    full,_=R.render_scene(sc,tex,bg,None,R.SceneParams(width=w,height=h,quality='full',margin_fade=0))
    preview,_=R.render_scene(sc,tex,bg,None,R.SceneParams(width=w,height=h,quality='preview',margin_fade=0))
    pole=lambda im:(im[...,0]>150)&(im[...,1]<80)
    assert pole(full).sum()>100
    assert pole(preview).sum() >= .9*pole(full).sum()


def test_original_body3d_keeps_real_arm_gaps(monkeypatch):
    import json,struct
    from lumen_engine.recon import body3d
    mask=np.zeros((100,80),np.float32);mask[5:95,10:70]=1
    mask[30:65,25:45]=0
    rgb=np.full((100,80,3),.5,np.float32)
    monkeypatch.setattr(body3d,'estimate_depth',lambda im,*a,**kw:np.full(im.shape[:2],.4,np.float32))
    out=body3d.build_body3d(rgb,mask,body3d.Body3DSettings(detail=80))
    blob=out['body'];hlen=struct.unpack_from('<I',blob,4)[0];hd=json.loads(blob[8:8+hlen]);n=hd['cols']*hd['rows']
    keep=np.frombuffer(blob,np.uint8,offset=8+hlen+4*n).reshape(hd['rows'],hd['cols'])
    assert keep[hd['rows']//2,hd['cols']//2] == 0
    assert keep.sum()>1000


def test_scene_pipeline_uses_sky_confidence_for_white_room(monkeypatch):
    from lumen_engine.recon import scene
    yy,xx=np.mgrid[:80,:120].astype(np.float32)
    d=np.where(yy<40,0,.2+.5*(yy-40)/40).astype(np.float32)
    rgb=np.full((80,120,3),(.92,.90,.88),np.float32)
    def depth(im,*args,info=None,**kw):
        if info is not None: info['sky_hint']=np.zeros(im.shape[:2],np.float32)
        return d.copy()
    monkeypatch.setattr(scene,'estimate_depth',depth)
    monkeypatch.setattr(scene,'inpaint',lambda image,*a,**kw:(image.copy(),'test'))
    out=scene.reconstruct(rgb,scene.ReconSettings(quality='fast',detail_fusion=False,remove_foreground=False,outpaint_margin=0))
    assert out['report']['sky_fraction']==0
    from lumen_engine.recon.render import parse_scene
    assert not np.any(parse_scene(out['scene']).disp[:30] == G.SKY_DISP)


def test_sparse_background_depth_is_not_misclassified_as_sky(monkeypatch):
    from lumen_engine.recon import render as R,environment
    sc,tm,tb,tk=R._synthetic_scene()
    sc.disp[:]=.2;sc.bg_disp[:]=-1
    captured=[]
    def environment_spy(d,*args,**kw):
        captured.append(d)
        return np.full((16,32,3),.5,np.float32)
    monkeypatch.setattr(environment,'scene_environment',environment_spy)
    R.scene_environment_texture(sc,tm,60,tb)
    assert np.array_equal(captured[0],sc.disp)


def test_detail_cap_does_not_erase_existing_fine_global_geometry(monkeypatch):
    from lumen_engine.depth import estimator as E
    from lumen_engine.models import manager
    h,w=900,720
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    rgb=np.stack([.4+.2*np.sin(xx*1.5), yy/h, np.zeros_like(xx)],-1).astype(np.float32)
    monkeypatch.setattr(manager,'require',lambda _: 'local-test-model')
    monkeypatch.setattr(E,'onnx_session',lambda _:None)
    monkeypatch.setattr(E,'_predict',lambda sess,img,size,tta:img[...,0]+img[...,1])
    monkeypatch.setattr(E,'refine_edges',lambda img,disp:disp)
    global_depth=E.estimate_depth(rgb,detail=False)
    fused=E.estimate_depth(rgb,detail=True,detail_side=800)
    assert np.abs(fused-global_depth).mean()<1e-4
