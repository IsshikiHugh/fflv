from fflv.format.vp9 import codec_profile, codec_string, split_superframe


def test_codec_string_levels():
    assert codec_string(1920, 1080, 30) == "vp09.00.40.08"
    assert codec_string(1920, 1080, 60) == "vp09.00.41.08"
    assert codec_string(1280, 720, 30) == "vp09.00.31.08"
    assert codec_string(640, 360, 30) == "vp09.00.21.08"
    assert codec_string(3840, 2160, 30) == "vp09.00.50.08"
    assert codec_string(64, 64, 30) == "vp09.00.10.08"


def test_codec_string_lossless_is_profile_1_rgb():
    assert codec_string(1280, 720, 30, lossless=True) == "vp09.01.31.08.03.01.13.00.01"
    assert codec_profile(codec_string(640, 360, 30, lossless=True)) == 1
    assert codec_profile("vp09.00.40.08") == 0
    assert codec_profile("avc1.42E01E") is None


def test_superframe_split():
    a, b = b"\x82\x01\x02", b"\x86\x05"
    # marker 0b110_00_001: 2 frames, 1-byte sizes
    marker = 0xC1
    data = a + b + bytes([marker, len(a), len(b), marker])
    frames, sf = split_superframe(data)
    assert sf and frames == [a, b]
    frames, sf = split_superframe(a)
    assert not sf and frames == [a]
