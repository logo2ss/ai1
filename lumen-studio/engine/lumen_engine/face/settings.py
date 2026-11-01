"""Tunable settings of the face replacement pipeline.

Two kinds of knobs live here:

* :class:`FaceSwapSettings` - the user-facing settings (sliders/toggles in the
  UI, ``settings`` in ``POST /api/face/swap``). Fields the client does not send
  keep these defaults.
* **Internal constants** of the pipeline (filter scales, clamps, strengths).
  Pipeline code reads them through :func:`tv` (``tv("relight_sigma", 0.09)``):
  the literal is the hand-chosen default, :data:`TUNED` holds values found by
  the data-driven optimisation in ``engine/tools/face_train`` (self-swap
  benchmark with ground truth), and the optimiser itself temporarily overrides
  them with :func:`set_overrides`. Constants nobody tuned keep their default,
  and a renamed/removed constant simply stops being overridden.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields  # noqa: F401


@dataclass
class FaceSwapSettings:
    blend_strength: float = 1.0  # weight of the gradient-domain (Poisson) correction
    color_match: float = 0.9  # 0..1 skin-tone adaptation to the target
    lighting_match: float = 0.7  # 0..1 transfer of target shading (direction, shadows)
    edge_refinement: float = 0.5  # 0..1 softness/edge-snapping of the transition
    preserve_source_shape: float = 0.2  # 0 = target proportions, 1 = source proportions
    mask_extent: float = 1.0  # 0.8..1.1 scales the replaced area (forehead/jaw coverage)
    keep_target_mouth: bool = True  # keep target teeth/tongue when its mouth is open
    match_grain: bool = True
    match_sharpness: bool = True
    force: bool = False  # proceed even if the inputs are judged incompatible
    # learned colour/lighting correction (face/learned.py, trained on the self-swap
    # benchmark); falls back to the hand-tuned transfer when its weights are missing
    learned_color: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> "FaceSwapSettings":
        s = cls()
        for k, v in (d or {}).items():
            if hasattr(s, k) and v is not None:
                cur = getattr(s, k)
                if isinstance(cur, bool):
                    v = v.strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)
                setattr(s, k, type(cur)(v))
        return s

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- internal constants
# Benchmark-optimised values of internal constants (see engine/tools/face_train/README.md).
# Keys are the names used in ``tv(name, default)`` calls; anything not listed keeps its code default.
TUNED: dict[str, float] = {}

_overrides: dict[str, float] = {}
_seen: dict[str, float] = {}


def tv(name: str, default: float) -> float:
    """Value of the internal constant ``name`` (override > tuned > ``default``)."""
    if name not in _seen:
        _seen[name] = default
    if name in _overrides:
        return _overrides[name]
    return TUNED.get(name, default)


def set_overrides(values: dict[str, float] | None) -> None:
    """Temporarily replace internal constants (used by the optimiser; not thread-safe)."""
    _overrides.clear()
    _overrides.update(values or {})


def seen_constants() -> dict[str, float]:
    """Internal constants read so far in this process, with their code defaults."""
    return dict(_seen)
