import json

import pytest


@pytest.fixture(scope="session")
def packed(tmp_path_factory):
    """A small but complete test file: 640x360, 4 s @ 30 fps, gop 30, all layer kinds, audio."""
    from fflv.devtools import testsrc
    from fflv.project import pack

    out = tmp_path_factory.mktemp("assets")
    assert testsrc.main(["--out", str(out), "--width", "640", "--height", "360", "--duration", "4", "--gop", "30"]) == 0
    rep = pack(out / "test_project.json", str(out / "small.lvd"), log=lambda s: None)
    assert rep.ok, [str(i) for i in rep.issues]
    return {
        "dir": out,
        "path": out / "small.lvd",
        "probes": json.loads((out / "barcodes.json").read_text()),
        "project": json.loads((out / "test_project.json").read_text()),
    }
