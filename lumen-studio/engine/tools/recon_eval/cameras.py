"""Standard camera set for novel-view evaluation (positions in scene units; the reference camera is at
the origin looking down -Z, scene depth runs from 1 (nearest) to ``zfar`` = 8 at the default settings)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Cam:
    name: str
    position: tuple = (0.0, 0.0, 0.0)
    yaw: float = 0.0  # degrees, + turns left (three.js Euler 'YXZ')
    pitch: float = 0.0  # degrees, + looks up
    look_at: tuple | None = None
    aspect: float | None = None  # output aspect; None = the photo's aspect
    group: str = ""  # lateral | vertical | walk | turn | wide | orbit | ref


# views shown in the index sheet (one row per scene)
KEY_VIEWS = ["ref", "orbitR5", "L0.50", "R0.50", "fwd1.00", "yawL25", "back0.50"]
# small camera moves (averaged separately as small_*)
SMALL_VIEWS = ["L0.25", "R0.25", "up0.25", "down0.25", "orbitL5", "orbitR5", "yawL10", "yawR10"]


def standard_cameras(pivot_z: float = 3.0) -> list[Cam]:
    """The standard set. ``pivot_z`` (median depth of the photo) is the orbit centre."""
    zp = float(min(max(pivot_z, 1.2), 7.0))
    th = math.asin(min(0.4 / zp, 0.6))
    ox, oz = zp * math.sin(th), -zp + zp * math.cos(th)
    s5, c5 = math.sin(math.radians(5.0)), math.cos(math.radians(5.0))
    views = [
        Cam("ref", group="ref"),
        Cam("L0.25", (-0.25, 0.0, 0.0), group="lateral"),
        Cam("R0.25", (0.25, 0.0, 0.0), group="lateral"),
        Cam("L0.50", (-0.5, 0.0, 0.0), group="lateral"),
        Cam("R0.50", (0.5, 0.0, 0.0), group="lateral"),
        Cam("up0.25", (0.0, 0.25, 0.0), group="vertical"),
        Cam("down0.25", (0.0, -0.25, 0.0), group="vertical"),
        Cam("fwd0.50", (0.0, 0.0, -0.5), group="walk"),
        Cam("fwd1.00", (0.0, 0.0, -1.0), group="walk"),
        Cam("yawL25", yaw=25.0, group="turn"),
        Cam("yawR25", yaw=-25.0, group="turn"),
        Cam("pitchU15", pitch=15.0, group="turn"),
        Cam("pitchD15", pitch=-15.0, group="turn"),
        Cam("back0.50", (0.0, 0.0, 0.5), group="wide"),
        Cam("wide", aspect=2.1, group="wide"),
        Cam("orbitL", (-ox, 0.0, oz), look_at=(0.0, 0.0, -zp), group="orbit"),
        Cam("orbitR", (ox, 0.0, oz), look_at=(0.0, 0.0, -zp), group="orbit"),
        # small moves (the user: "turns into rubbish when you turn ever so slightly"): a 5 deg orbit drag
        # around the median depth as the viewer does (= ~0.29 sideways at zp 3.3), small turns in place
        Cam("orbitL5", (-zp * s5, 0.0, -zp + zp * c5), look_at=(0.0, 0.0, -zp), group="small"),
        Cam("orbitR5", (zp * s5, 0.0, -zp + zp * c5), look_at=(0.0, 0.0, -zp), group="small"),
        Cam("yawL10", yaw=10.0, group="small"),
        Cam("yawR10", yaw=-10.0, group="small"),
    ]
    for degrees in (35, 50):
        for sign, side in ((-1, "L"), (1, "R")):
            a = math.radians(sign*degrees)
            views.append(Cam(f"orbit{side}{degrees}", (zp*math.sin(a), 0.0, -zp+zp*math.cos(a)), look_at=(0.0, 0.0, -zp), group="orbit"))
    return views
