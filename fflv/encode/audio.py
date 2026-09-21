"""Opus audio track: transcode with FFmpeg, read packets with PyAV (spec 8.2 step 4).

Packets are stored on the decoder timeline (first packet at 0); the Opus pre-skip goes into the
metadata (LVF_SPEC.md appendix A.4).
"""

from __future__ import annotations

import base64
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av

from ..format.binary import AudioPacket
from ..format.constants import OPUS_SAMPLE_RATE
from .media import MediaError, run


@dataclass
class AudioTrack:
    packets: list[AudioPacket]
    extradata: bytes
    channels: int
    pre_skip: int

    def meta(self) -> dict:
        return {"codec": "opus", "sample_rate": OPUS_SAMPLE_RATE, "channels": self.channels,
                "description_b64": base64.b64encode(self.extradata).decode("ascii"), "pre_skip": self.pre_skip}

    def packets_before(self, end_us: int) -> list[AudioPacket]:
        return [p for p in self.packets if p.pts_us < end_us]


def samples_to_us(samples: int) -> int:
    return (samples * 2_000_000 + OPUS_SAMPLE_RATE) // (2 * OPUS_SAMPLE_RATE)


def encode_audio(src: str | Path, *, max_seconds: float | None = None, bitrate: str = "128k",
                 channels: int = 2) -> AudioTrack:
    """Transcode any audio source to 48 kHz Opus and return its packets."""
    src = Path(src)
    if not src.exists():
        raise MediaError(f"audio source not found: {src}")
    with tempfile.TemporaryDirectory(prefix="fflv_audio_") as tmp:
        out = Path(tmp) / "audio.ogg"
        limit = ["-t", f"{max_seconds:.6f}"] if max_seconds is not None else []
        run(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
             "-map", "0:a:0", "-vn", "-sn", "-dn", "-c:a", "libopus", "-b:a", bitrate,
             "-ar", str(OPUS_SAMPLE_RATE), "-ac", str(channels), *limit, "-f", "ogg", str(out)],
            f"audio transcode of {src.name}")
        packets: list[AudioPacket] = []
        with av.open(str(out)) as c:
            st = c.streams.audio[0]
            extradata = bytes(st.codec_context.extradata or b"")
            if extradata[:8] != b"OpusHead" or len(extradata) < 19:
                raise MediaError("Opus stream has no OpusHead extradata")
            if st.time_base != Fraction(1, OPUS_SAMPLE_RATE):
                raise MediaError(f"unexpected Opus time base {st.time_base}")
            pre_skip = int.from_bytes(extradata[10:12], "little")
            for p in c.demux(st):
                if not p.size or p.pts is None:
                    continue
                t = p.pts + pre_skip  # decoder timeline: first packet at 0
                if t < 0:
                    raise MediaError(f"Opus packet before the stream start (pts {p.pts})")
                packets.append(AudioPacket(samples_to_us(t), samples_to_us(p.duration or 0), bytes(p)))
    return AudioTrack(packets, extradata, extradata[9], pre_skip)


def audio_from_meta(meta_audio: dict, packets: list[AudioPacket]) -> AudioTrack:
    extradata = base64.b64decode(meta_audio["description_b64"]) if meta_audio.get("description_b64") else b""
    return AudioTrack(packets, extradata, meta_audio["channels"], meta_audio.get("pre_skip", 0))
