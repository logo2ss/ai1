"""Novel-view evaluation harness for the single-photo 3D reconstruction.

Run from ``engine/``::

    python -m tools.recon_eval --scenes demo03,demo07 --tag mytag [--quality fast] [--max-side 800]
    python -m tools.recon_eval --scenes std --tag mytag          # the recommended scene set
    python -m tools.recon_eval --compare base,mytag             # side-by-side sheets + metric diff

Each scene is reconstructed once (cached under ``/tmp/claude-0/recon_eval/<tag>/<scene>/``), then a
standard camera set is rendered in-process with the CPU renderer (``lumen_engine.recon.render``).
Metrics are computed from the renderer's G-buffer (which layer every pixel shows, its texture
coordinates) and from the scene container itself; see ``metrics.py`` and the documentation in
``/tmp/claude-0/coord/recon_eval.md``.
"""

OUT_ROOT = "/tmp/claude-0/recon_eval"

# Recommended diverse scene set (people indoors/outdoors, street, rooms, landscape, buildings, objects).
STANDARD_SCENES = [
    "demo01",  # London street: people, bus, cars, trees, sky
    "demo16",  # people with bicycles outdoors, backlit, long shadows
    "faces/2008_002079",  # people indoors at a restaurant table (room behind them)
    "demo10",  # room interior with windows and furniture
    "demo15",  # palace interior: columns, staircase, small people
    "demo02",  # landscape: sunflowers in front of a cloudy sky
    "home",  # building facade against the sky (birds)
    "demo11",  # many cats (separate foreground objects) on a street
]
