"""Argument handling shared by the Python API: images, rects, frame rates, layer keys."""

from __future__ import annotations

import numbers
import os
import re
from fractions import Fraction

import numpy as np

from . import _fflv
from ._fflv import MediaError, MetaError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def as_uint8(image, name: str = "image") -> np.ndarray:
    """A C-contiguous uint8 array shaped H×W, H×W×3 or H×W×4.

    Accepts uint8 / bool / float (0..1) arrays shaped H×W (gray), H×W×1, H×W×3 (RGB) or H×W×4
    (RGBA). Images without alpha are opaque.
    """
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    elif np.issubdtype(a.dtype, np.floating):
        a = np.clip(np.nan_to_num(a) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        raise MetaError(f"{name}: dtype {a.dtype} is not supported (use uint8, bool or float in 0..1)")
    if not (a.ndim == 2 or (a.ndim == 3 and a.shape[2] in (3, 4))):
        raise MetaError(f"{name}: expected H×W, H×W×3 or H×W×4, got shape {a.shape}")
    return np.ascontiguousarray(a)


def still_png(image) -> bytes:
    """PNG bytes for a still layer: PNG bytes as they are, an image file (converted to PNG if it is
    not one), or an array."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        if not data.startswith(PNG_SIGNATURE):
            raise MediaError("still image bytes are not a PNG")
        return data
    if isinstance(image, (str, os.PathLike)):
        return _fflv.png_from_file(os.fspath(image))
    return _fflv.encode_png(as_uint8(image, "still image"))


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def rect(value) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) from a tuple / list / array, a dict with x, y, w, h, or "x,y,w,h"."""
    if value is None:
        return None
    v = value
    if isinstance(v, dict):
        v = [v.get(k) for k in ("x", "y", "w", "h")]
    elif isinstance(v, str):
        parts = [s for s in re.split(r"[,x: ]+", v.strip()) if s]
        try:
            v = [int(p) for p in parts]
        except ValueError:
            raise MetaError(f"rect must be four integers x,y,w,h, got {value!r}") from None
    else:
        try:
            v = list(v)
        except TypeError:
            raise MetaError(f"rect must be four integers [x, y, w, h], got {value!r}") from None
    if len(v) != 4 or not all(_is_int(n) for n in v):
        raise MetaError(f"rect must be four integers [x, y, w, h], got {value!r}")
    x, y, w, h = (int(n) for n in v)
    if w <= 0 or h <= 0:
        raise MetaError(f"rect width/height must be positive, got {value!r}")
    return x, y, w, h


def fps(value) -> Fraction:
    """Accepts 30, 30.0, "30/1", "30000/1001", "29.97", a Fraction or {"num":..,"den":..}."""
    try:
        if isinstance(value, dict):
            f = Fraction(int(value["num"]), int(value["den"]))
        elif isinstance(value, Fraction):
            f = value
        elif isinstance(value, (numbers.Real, str)) and not isinstance(value, bool):
            f = Fraction(str(value).strip())
        else:
            raise ValueError
    except (ValueError, KeyError, ZeroDivisionError):
        raise MetaError(f"cannot parse fps {value!r}") from None
    if f <= 0:
        raise MetaError(f"fps must be positive, got {value!r}")
    return f


def number(value, what: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, str)):
        raise MetaError(f"{what} must be a number, got {value!r}")
    try:
        return float(value)
    except ValueError:
        raise MetaError(f"{what} must be a number, got {value!r}") from None


def key(k) -> str:
    """A layer id or index, as the Rust side takes it."""
    if _is_int(k):
        return str(int(k))
    if isinstance(k, str):
        return k
    raise MetaError(f"a layer is an id or an index, got {k!r}")


def jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (tuple, list, np.ndarray)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    return v


def frame_arg(v, what: str) -> int | None:
    if v is None:
        return None
    if not _is_int(v) or v < 0:
        raise MetaError(f"{what} must be a frame number (non-negative integer), got {v!r}")
    return int(v)
