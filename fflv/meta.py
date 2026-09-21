"""Building and checking metadata (spec section 4) for the writer, `pack` and the edit commands."""

from __future__ import annotations

import re
from fractions import Fraction

from .format.constants import ALPHA_RANGES, BLEND_MODES
from .format.vp9 import codec_string

GENERATOR = "fflv"
_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
EDITABLE_FIELDS = ("id", "name", "z", "rect", "blend", "opacity", "visible")


class MetaError(ValueError):
    pass


def check_id(layer_id: str, taken: set[str] = frozenset()) -> str:
    if not isinstance(layer_id, str) or not _ID.match(layer_id):
        raise MetaError(f"layer id {layer_id!r} must be letters, digits, '_', '-', '.' (not starting with '-' or '.')")
    if layer_id.isdigit():
        raise MetaError(f"layer id {layer_id!r} must not be a plain number (numbers refer to layer indices)")
    if layer_id in taken:
        raise MetaError(f"layer id {layer_id!r} is already used")
    return layer_id


def check_rect(rect) -> dict:
    if isinstance(rect, dict):
        rect = [rect.get(k) for k in ("x", "y", "w", "h")]
    if isinstance(rect, str):
        rect = [s for s in re.split(r"[,x: ]+", rect.strip()) if s]
        try:
            rect = [int(v) for v in rect]
        except ValueError:
            raise MetaError(f"rect must be four integers x,y,w,h, got {rect!r}") from None
    if not isinstance(rect, (list, tuple)) or len(rect) != 4 or not all(isinstance(v, int) for v in rect):
        raise MetaError(f"rect must be four integers [x, y, w, h], got {rect!r}")
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        raise MetaError(f"rect width/height must be positive, got {rect!r}")
    return {"x": x, "y": y, "w": w, "h": h}


def check_blend(blend: str) -> str:
    if blend not in BLEND_MODES:
        raise MetaError(f"blend must be one of {BLEND_MODES}, got {blend!r}")
    return blend


def check_opacity(opacity) -> float:
    try:
        v = float(opacity)
    except (TypeError, ValueError):
        raise MetaError(f"opacity must be a number, got {opacity!r}") from None
    if not 0.0 <= v <= 1.0:
        raise MetaError(f"opacity must be in [0, 1], got {v}")
    return v


def check_background(color: str) -> str:
    if not isinstance(color, str) or not _HEX.match(color):
        raise MetaError(f"background must be #RRGGBB, got {color!r}")
    return color


def parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise MetaError(f"expected true/false, got {v!r}")


def even(n: int) -> int:
    return n + (n & 1)


def video_layer(*, id: str, name: str, z: float, rect: dict, start: int, end: int, fps: Fraction,
                alpha: bool, lossless: bool, blend: str = "normal", opacity: float = 1.0,
                visible: bool = True) -> dict:
    cw, ch = even(rect["w"]), even(rect["h"])
    cs = codec_string(cw, ch, float(fps), lossless=lossless)
    d = {"id": id, "name": name, "kind": "video", "z": z, "rect": dict(rect),
         "start_frame": start, "end_frame": end,
         "codec": cs, "coded_width": cw, "coded_height": ch,
         "has_alpha": alpha, "alpha_codec": codec_string(cw, ch, float(fps)) if alpha else None}
    if (cw, ch) != (rect["w"], rect["h"]):
        # odd sizes are padded to even for 4:2:0; players crop back to the content
        d["content_size"] = [rect["w"], rect["h"]]
    if lossless:
        d["lossless"] = True
    if alpha:
        d["alpha_range"] = "full" if lossless else "limited"
    d.update({"blend": blend, "opacity": opacity, "visible": visible})
    assert d.get("alpha_range", "limited") in ALPHA_RANGES
    return d


def still_layer(*, id: str, name: str, z: float, rect: dict, start: int, end: int, offset: int, length: int,
                blend: str = "normal", opacity: float = 1.0, visible: bool = True) -> dict:
    return {"id": id, "name": name, "kind": "still", "z": z, "rect": dict(rect),
            "start_frame": start, "end_frame": end,
            "resource": {"offset": offset, "length": length, "mime": "image/png"},
            "blend": blend, "opacity": opacity, "visible": visible}


def file_meta(*, width: int, height: int, background: str, fps: Fraction, frame_count: int, gop: int,
              layers: list[dict], audio: dict | None) -> dict:
    return {
        "format": "LVF", "version": 1, "generator": GENERATOR,
        "canvas": {"width": width, "height": height, "background": background},
        "fps": {"num": fps.numerator, "den": fps.denominator},
        "frame_count": frame_count,
        "max_rap_interval": gop,
        "layers": layers,
        "audio": audio,
    }


def resolve_layer(meta: dict, key) -> int:
    """Layer index from an id or an index (int or numeric string)."""
    layers = meta["layers"]
    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        i = int(key)
        if not 0 <= i < len(layers):
            raise MetaError(f"layer index {i} out of range (file has {len(layers)} layers)")
        return i
    for i, L in enumerate(layers):
        if L["id"] == key:
            return i
    raise MetaError(f"no layer {key!r}; layers: {', '.join(L['id'] for L in layers)}")


def apply_edits(meta: dict, key, fields: dict) -> dict:
    """Validate and apply editable field changes to one layer (returns the layer dict)."""
    i = resolve_layer(meta, key)
    L = meta["layers"][i]
    for k, v in fields.items():
        if k not in EDITABLE_FIELDS:
            raise MetaError(f"field {k!r} cannot be edited in place (editable: {', '.join(EDITABLE_FIELDS)}); "
                            f"frame ranges and pixels need `fflv rm` + `fflv add`")
        if k == "id":
            L["id"] = check_id(v, {x["id"] for j, x in enumerate(meta["layers"]) if j != i})
        elif k == "name":
            L["name"] = str(v)
        elif k == "z":
            try:
                L["z"] = int(v) if str(v).lstrip("-").isdigit() else float(v)
            except ValueError:
                raise MetaError(f"z must be a number, got {v!r}") from None
        elif k == "rect":
            L["rect"] = check_rect(v)
        elif k == "blend":
            L["blend"] = check_blend(v)
        elif k == "opacity":
            L["opacity"] = check_opacity(v)
        elif k == "visible":
            L["visible"] = parse_bool(v)
    return L
