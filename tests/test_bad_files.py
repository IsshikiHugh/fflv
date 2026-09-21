"""Acceptance 11.2-2: deliberately broken files are reported precisely."""

import pytest

from fflv.devtools import corrupt as make_bad_files


@pytest.mark.parametrize("mutation", make_bad_files.MUTATIONS, ids=lambda m: m.name)
def test_validator_reports_broken_file(packed, tmp_path, mutation):
    path = make_bad_files.write_variant(str(packed["path"]), mutation, tmp_path)
    ok, msg = make_bad_files.check(path, mutation)
    assert ok, msg


def test_truncated_file_is_reported(packed, tmp_path):
    from fflv.format import validate
    data = packed["path"].read_bytes()
    bad = tmp_path / "truncated.lvd"
    bad.write_bytes(data[: len(data) // 2])
    rep = validate(str(bad))
    assert not rep.ok
    assert rep.error_codes() & {"HDR", "CAU", "I1", "I10"}


def test_garbage_is_reported(tmp_path):
    from fflv.format import validate
    bad = tmp_path / "garbage.lvd"
    bad.write_bytes(b"NOPE" + bytes(100))
    rep = validate(str(bad))
    assert rep.fatal and "HDR" in rep.error_codes()
