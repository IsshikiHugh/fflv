"""fflv — pack, edit, decode, render and view LVF layered video files (.lvd).

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

__version__ = "0.1.0"

from .decode import Reader, open  # noqa: E402,A004
from .edit import add_layer, add_still, remove_layers, set_audio, set_layer  # noqa: E402
from .encode.writer import Writer  # noqa: E402
from .format import validate  # noqa: E402
from .project import pack  # noqa: E402
from .render import extract, render  # noqa: E402

__all__ = ["Writer", "Reader", "open", "add_layer", "add_still", "remove_layers", "set_audio", "set_layer",
           "validate", "pack", "render", "extract", "__version__"]
