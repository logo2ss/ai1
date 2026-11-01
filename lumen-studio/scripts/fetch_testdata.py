#!/usr/bin/env python3
"""Download the public test images used by the automated tests (optional)."""

from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / "testdata"
GH = "https://raw.githubusercontent.com"
FILES = {
    "faces": [f"{GH}/ageitgey/face_recognition/master/examples/{f}" for f in ["obama.jpg", "biden.jpg", "obama2.jpg", "obama_small.jpg", "two_people.jpg", "alex-lacamoire.png", "lin-manuel-miranda.png"]]
    + [f"{GH}/davisking/dlib/master/examples/faces/{f}" for f in ["2007_007763.jpg", "2008_001009.jpg", "2008_001322.jpg", "2008_002079.jpg", "2008_002470.jpg", "2008_002506.jpg", "2008_004176.jpg", "2009_004587.jpg", "bald_guys.jpg"]]
    + [f"{GH}/opencv/opencv/4.x/samples/data/{f}" for f in ["lena.jpg", "messi5.jpg"]],
    "scenes": [f"{GH}/DepthAnything/Depth-Anything-V2/main/assets/examples/demo{i:02d}.jpg" for i in range(1, 21)]
    + [f"{GH}/opencv/opencv/4.x/samples/data/{f}" for f in ["building.jpg", "home.jpg", "fruits.jpg"]],
}

for sub, urls in FILES.items():
    d = ROOT / sub
    d.mkdir(parents=True, exist_ok=True)
    for u in urls:
        p = d / u.rsplit("/", 1)[1]
        if not p.exists():
            print("↓", p.name)
            urllib.request.urlretrieve(u, p)
print("done:", ROOT)
