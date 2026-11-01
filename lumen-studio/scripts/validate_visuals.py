"""Offline real-photo contact sheets for manual quality review (run from project root)."""
from pathlib import Path
import json
import math
import time
import cv2
import numpy as np
from lumen_engine.face import backend
from lumen_engine.face.swap import swap_face, FaceSwapSettings
from lumen_engine.imaging.io import decode_image, resize_max
from lumen_engine.recon.person3d import build_person3d, Person3DSettings
from lumen_engine.recon.person_render import load_person, render_person

OUT=Path('validation'); OUT.mkdir(exist_ok=True)
def load(p, limit=1000): return resize_max(decode_image(Path('testdata', p).read_bytes())[0],limit)[0]
def save(p, rgb): cv2.imwrite(str(OUT/p),cv2.cvtColor((np.clip(rgb,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
def faces():
    src,tgt=load('faces/biden.jpg'),load('faces/obama.jpg')
    fs,ft=backend.detect(src)[0][0],backend.detect(tgt)[0][0]
    reports={}
    for mode in ('pure','natural'):
        start=time.perf_counter()
        out,rep,_=swap_face(src,fs,tgt,ft,FaceSwapSettings(mode=mode,force=True))
        reports[mode]={'seconds':time.perf_counter()-start,'report':rep}
        save(mode+'-swap.jpg',np.concatenate([tgt,out],1))
    (OUT/'face-reports.json').write_text(json.dumps(reports,indent=2))
def person():
    rgb=load('faces/obama.jpg'); out=build_person3d(rgb,settings=Person3DSettings(detail=220))
    model=load_person(out['person'])
    tex=cv2.cvtColor(cv2.imdecode(np.frombuffer(out['texture'],np.uint8),1),cv2.COLOR_BGR2RGB)
    frames=[]
    for deg in (-50,0,50):
        a=math.radians(deg)
        frame,_=render_person(model,tex,(2.3*math.sin(a),0,2.3*math.cos(a)),width=480,height=560)
        frames.append(frame.astype(np.float32)/255)
    save('person-angles.jpg',np.concatenate(frames,1))
def regions():
    from lumen_engine.region.swap import insert_region, RegionSwapSettings
    # Real sports-photo pair: the selected football replaces the basketball.
    src=load('faces/messi5.jpg',600); dst=load('scenes/demo07.jpg',600)
    ma=np.zeros(src.shape[:2],np.float32);cv2.circle(ma,(362,313),24,1,-1)
    mb=np.zeros(dst.shape[:2],np.float32)
    cv2.ellipse(mb,(round(dst.shape[1]*0.40),round(dst.shape[0]*0.34)),(round(dst.shape[1]*0.13),round(dst.shape[0]*0.18)),0,0,360,1,-1)
    out,rep,dbg=insert_region(src,ma,dst,mb,RegionSwapSettings(mode='object',fit='contain',extract_object=False,snap_edges=True))
    save('region-object.jpg',np.concatenate([dst,out],1)); save('region-plate.jpg',dbg['plate'])
    # Sky texture between two outdoor photographs (area fit / seam test).
    sky=load('scenes/demo02.jpg',600); landscape=load('scenes/demo09.jpg',600)
    a=np.zeros(sky.shape[:2],np.float32);a[:round(len(a)*0.14)]=1
    b=np.zeros(landscape.shape[:2],np.float32);b[:round(len(b)*0.30)]=1
    area,ar,_=insert_region(sky,a,landscape,b,RegionSwapSettings(mode='area',fit='cover',snap_edges=False))
    save('region-area.jpg',np.concatenate([landscape,area],1))
    (OUT/'region-reports.json').write_text(json.dumps({'object':rep,'area':ar},indent=2))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('tool',choices=['faces','person','regions']);a=ap.parse_args()
    {'faces':faces,'person':person,'regions':regions}[a.tool]()
