"""Image decoding/encoding with correct orientation, alpha and bit depth.

Internally the engine works with ``float32`` RGB arrays in ``[0, 1]``
(``H x W x 3``) and separate ``float32`` masks in ``[0, 1]`` (``H x W``).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

from ..errors import InvalidInputError

Image.MAX_IMAGE_PIXELS = 300_000_000  # allow large photos, but not decompression bombs
MAX_PIXELS = 120_000_000


def decode_image(data: bytes) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Decode bytes to (rgb float32, alpha or None, info)."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:
        raise InvalidInputError(
            "The file could not be read as an image.",
            hint="Supported formats: JPEG, PNG, WebP, TIFF, BMP.",
        ) from exc
    if im.width * im.height > MAX_PIXELS:
        raise InvalidInputError(f"The image is too large ({im.width}x{im.height}).")
    info = {"format": im.format, "mode": im.mode, "icc_profile": im.info.get("icc_profile")}
    if im.format in ("PNG", "TIFF"):
        # Pillow truncates 16-bit RGB(A); decode those with OpenCV at full depth
        import cv2

        deep = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if deep is not None and deep.dtype == np.uint16 and deep.ndim == 3:
            arr = deep.astype(np.float32) / 65535.0
            rgb = cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2RGB)
            alpha = arr[..., 3].copy() if arr.shape[2] == 4 else None
            info["deep"] = True
            return np.ascontiguousarray(rgb), alpha, info
    im = ImageOps.exif_transpose(im)
    alpha = None
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float32)
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
        rgb = np.repeat(arr[..., None], 3, axis=2)
        return np.clip(rgb, 0, 1).astype(np.float32), None, info
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        arr = np.asarray(im, dtype=np.float32) / 255.0
        alpha = arr[..., 3].copy()
        if alpha.min() > 0.999:
            alpha = None
        rgb = arr[..., :3]
    else:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(rgb), alpha, info


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def encode_image(
    rgb: np.ndarray,
    fmt: str = "png",
    quality: int = 95,
    alpha: np.ndarray | None = None,
    icc_profile: bytes | None = None,
    text: dict | None = None,
    bit_depth: int = 8,
) -> bytes:
    fmt = fmt.lower()
    buf = io.BytesIO()
    if bit_depth == 16 and fmt in ("png", "tiff", "tif"):
        arr16 = (np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16)
        import cv2

        ok, enc = cv2.imencode("." + ("png" if fmt == "png" else "tiff"), cv2.cvtColor(arr16, cv2.COLOR_RGB2BGR))
        if not ok:
            raise InvalidInputError("16-bit encoding failed")
        return enc.tobytes()
    arr = to_uint8(rgb)
    if alpha is not None and fmt in ("png", "webp", "tiff", "tif"):
        arr = np.dstack([arr, to_uint8(alpha)])
    im = Image.fromarray(arr)
    kwargs: dict = {}
    if icc_profile:
        kwargs["icc_profile"] = icc_profile
    if fmt in ("jpg", "jpeg"):
        im.convert("RGB").save(buf, "JPEG", quality=quality, subsampling=0, optimize=True, **kwargs)
    elif fmt == "webp":
        im.save(buf, "WEBP", quality=quality, method=5, **kwargs)
    elif fmt in ("tif", "tiff"):
        im.save(buf, "TIFF", compression="tiff_lzw", **kwargs)
    else:
        info = PngImagePlugin.PngInfo()
        for k, v in (text or {}).items():
            info.add_text(k, v)
        im.save(buf, "PNG", pnginfo=info, compress_level=6 if arr.size < 4e7 else 3, **kwargs)
    return buf.getvalue()


def decode_mask(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Decode a mask image (white = selected) and resize it to ``shape`` (h, w)."""
    import cv2

    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:
        raise InvalidInputError("The selection mask could not be decoded.") from exc
    if im.mode in ("RGBA", "LA"):
        arr = np.asarray(im.convert("RGBA"), dtype=np.float32)
        # Selection drawn as coloured overlay: use alpha
        m = arr[..., 3] / 255.0
    else:
        m = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    if m.shape != tuple(shape):
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(m, 0, 1).astype(np.float32)


def resize_max(img: np.ndarray, max_side: int, interp=None) -> tuple[np.ndarray, float]:
    import cv2

    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img, 1.0
    interp = cv2.INTER_AREA if interp is None else interp
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp), s
