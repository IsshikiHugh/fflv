"""fflv — pack, edit, decode, render and view LVF layered video files (.lvd).

The work is done by the Rust core (the `fflv._fflv` extension); encoding and decoding run in
parallel without holding the GIL.

Writing from code::

    import fflv
    with fflv.Writer("debug.lvd", size=(1280, 720), fps=30) as w:
        w.add_layer("frame")
        w.add_layer("mask", alpha=True, lossless=True)
        for img, mask in data:
            w.write(frame=img, mask=mask)

Reading::

    with fflv.open("debug.lvd") as f:
        rgb = f.frame(120, layers=["frame", "mask"])

Editing (existing layers are never re-encoded)::

    fflv.add_layer("debug.lvd", "pred", predictions, alpha=True)
    fflv.remove_layers("debug.lvd", ["pred"])
    fflv.set_layer("debug.lvd", "mask", opacity=0.5)
"""

from . import cli
from ._fflv import (DecodeError, EditError, EncodeError, FormatError, InvalidOutput, MediaError, MetaError,
                    OutputError, PackError, ViewError, WriterError, __version__)
from .edit import add_layer, add_still, remove_layers, set_audio, set_layer
from .reader import LayerInfo, Reader, open  # noqa: A004
from .report import Issue, LayerStats, Report
from .tools import extract, pack, render, validate, view
from .writer import Writer

__all__ = [
    "Writer", "Reader", "open", "LayerInfo", "add_layer", "add_still", "remove_layers", "set_audio", "set_layer",
    "validate", "pack", "render", "extract", "view", "Report", "Issue", "LayerStats",
    "MetaError", "FormatError", "OutputError", "EncodeError", "DecodeError", "MediaError", "WriterError",
    "EditError", "PackError", "ViewError", "InvalidOutput", "__version__",
]
