"""`fflv pack`: build an .lvd from a project JSON (spec section 8.1).

Pipeline (spec 8.2): transcode every video layer to VP9 IVF with FFmpeg (color and alpha planes, in
parallel), verify every packet ("do not trust the encoder"), transcode audio to Opus, interleave
everything into composite frames, write index + header, validate.

Project additions over spec 8.1: per layer `lossless` (bit-exact RGB / alpha), `start_frame` /
`end_frame` instead of seconds; `quality.alpha_crf`, `quality.cpu_used`; `audio.bitrate`,
`audio.channels`.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av

from . import meta as M
from .encode.audio import AudioTrack, encode_audio
from .encode.media import MediaError, VideoSource, input_args, layer_filter, open_video_source, run, still_png
from .format import Cau, LVFWriter, Report, VideoEntry, parse_fps, pts_us, seconds_to_frame
from .format.output import InvalidOutput, publish, temp_path_for
from .format.constants import CAU_FLAG_RAP, ENTRY_EMPTY, ENTRY_FRAME, FILE_EXTENSION, FRAME_FLAG_KEY
from .format.vp9 import Vp9Error, inspect_packet


class PackError(Exception):
    pass


Log = Callable[[str], None]


@dataclass
class Layer:
    index: int
    id: str
    name: str
    kind: str
    src: Path
    z: float
    rect: dict
    start_frame: int
    end_frame: int
    alpha: bool
    lossless: bool
    blend: str
    opacity: float
    visible: bool
    source: VideoSource | None = None
    color_ivf: Path | None = None
    alpha_ivf: Path | None = None
    keyflags: list[bool] = field(default_factory=list)
    png: bytes = b""

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def coded(self) -> tuple[int, int]:
        return M.even(self.rect["w"]), M.even(self.rect["h"])


@dataclass
class Project:
    path: Path
    output: Path
    width: int
    height: int
    background: str
    fps: Fraction
    frame_count: int
    gop: int
    crf: int
    alpha_crf: int
    cpu_used: int
    layers: list[Layer]
    audio_src: Path | None
    audio_bitrate: str
    audio_channels: int


def _frame(L: dict, key: str, fps: Fraction, default: int) -> int:
    if f"{key}_frame" in L:
        return int(L[f"{key}_frame"])
    if L.get(key) is None:
        return default
    return seconds_to_frame(L[key], fps)


def load_project(path: Path, output: str | None = None) -> Project:
    try:
        P = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackError(f"cannot read project {path}: {exc}") from exc
    base = path.parent

    def src_path(s) -> Path:
        if not isinstance(s, str) or not s:
            raise PackError(f"src must be a path string, got {s!r}")
        p = Path(s)
        p = p if p.is_absolute() else base / p
        if not p.exists():
            raise PackError(f"source file not found: {p}")
        return p

    try:
        canvas = P.get("canvas") or {}
        width, height = canvas.get("width"), canvas.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise PackError(f"canvas needs positive integer width/height, got {canvas!r}")
        fps = parse_fps(P.get("fps", "30/1"))
        if "duration" not in P:
            raise PackError("project needs a duration (seconds)")
        frame_count = seconds_to_frame(P["duration"], fps)
        if frame_count <= 0:
            raise PackError(f"duration {P['duration']!r} gives {frame_count} frames")
        gop = int(P.get("gop", round(2 * fps)))
        if gop <= 0:
            raise PackError("gop must be positive")
        q = P.get("quality") or {}
        crf = int(q.get("crf", 32))

        layers: list[Layer] = []
        ids: set[str] = set()
        for i, L in enumerate(P.get("layers") or []):
            tag = f"layer {i} ({L.get('id')!r})"
            lid = M.check_id(L.get("id"), ids)
            ids.add(lid)
            kind = L.get("kind", "video")
            if kind not in ("video", "still"):
                raise PackError(f"{tag}: kind must be video or still")
            start = _frame(L, "start", fps, 0)
            end = min(_frame(L, "end", fps, frame_count), frame_count)
            if not 0 <= start < end:
                raise PackError(f"{tag}: empty or negative interval [{start}, {end}) (frame_count {frame_count})")
            layers.append(Layer(
                index=i, id=lid, name=str(L.get("name", lid)), kind=kind, src=src_path(L.get("src")),
                z=L.get("z", i), rect=M.check_rect(L.get("rect")), start_frame=start, end_frame=end,
                alpha=bool(L.get("alpha", False)), lossless=bool(L.get("lossless", False)),
                blend=M.check_blend(L.get("blend", "normal")), opacity=M.check_opacity(L.get("opacity", 1.0)),
                visible=bool(L.get("visible", True)),
            ))
        if not layers:
            raise PackError("project has no layers")
        background = M.check_background(canvas.get("background", "#000000"))
    except M.MetaError as exc:
        raise PackError(str(exc)) from exc

    audio = P.get("audio")
    out = output or P.get("output") or path.with_suffix(FILE_EXTENSION).name
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = (Path.cwd() if output else base) / out_path
    return Project(
        path=path, output=out_path, width=width, height=height, background=background, fps=fps,
        frame_count=frame_count, gop=gop, crf=crf, alpha_crf=int(q.get("alpha_crf", crf)),
        cpu_used=int(q.get("cpu_used", 4)), layers=layers,
        audio_src=src_path(audio["src"]) if audio else None,
        audio_bitrate=str((audio or {}).get("bitrate", "128k")),
        audio_channels=int((audio or {}).get("channels", 2)),
    )


# --------------------------------------------------------------------------------------------------
# Transcoding with FFmpeg
# --------------------------------------------------------------------------------------------------
def prepare_video_layer(L: Layer) -> None:
    L.source = open_video_source(L.src)
    if L.alpha and not L.source.has_alpha:
        raise PackError(f"layer {L.id!r}: alpha=true but {L.src.name} ({L.source.codec}, {L.source.pix_fmt}) "
                        f"has no alpha channel")


def vp9_cmd(proj: Project, L: Layer, plane: str, out: Path) -> list[str]:
    N, G = L.n_frames, proj.gop
    vf = layer_filter(proj.fps, L.rect["w"], L.rect["h"])
    color_meta = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    if plane == "color" and L.lossless:
        # profile 1, 8-bit 4:4:4 RGB, lossless: pixel values survive exactly
        vf += ["format=gbrp"]
        pix, meta_args = "gbrp", ["-colorspace", "rgb", "-color_primaries", "bt709",
                                  "-color_trc", "iec61966-2-1", "-color_range", "pc"]
    elif plane == "color":
        vf += ["scale=out_color_matrix=bt709:out_range=limited", "format=yuv420p"]
        pix, meta_args = "yuv420p", color_meta + ["-color_range", "tv"]
    else:
        # alpha → luma. Lossy: full-range alpha becomes limited-range Y 16..235, as the VP9 header
        # says (color_range=0). Lossless: Y = alpha exactly, header says full range.
        rng = "full" if L.lossless else "limited"
        vf += ["format=rgba", "alphaextract", f"scale=in_range=full:out_range={rng}", "format=yuv420p"]
        pix, meta_args = "yuv420p", color_meta + ["-color_range", "pc" if L.lossless else "tv"]
    vf.append("tpad=stop_mode=clone:stop=-1")
    # Key frames: local frame 0 (I4) and every frame on the global GOP grid. libvpx's own interval
    # is pushed past the clip (it does not restart after a forced key frame).
    kf = f"expr:eq(n,0)+eq(mod(n+{L.start_frame % G},{G}),0)"
    quality = ["-lossless", "1"] if L.lossless else ["-crf", str(proj.crf if plane == "color" else proj.alpha_crf)]
    return ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *input_args(L.source),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", ",".join(vf), "-frames:v", str(N),
            "-fps_mode", "passthrough", "-c:v", "libvpx-vp9", "-pix_fmt", pix,
            *quality, "-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", str(proj.cpu_used),
            "-g", str(N + G), "-keyint_min", str(N + G), "-force_key_frames", kf,
            "-auto-alt-ref", "0", "-lag-in-frames", "0", *meta_args, "-f", "ivf", str(out)]


def iter_ivf(path: Path):
    with av.open(str(path)) as c:
        for p in c.demux(video=0):
            if p.size:
                yield p


def scan_plane(L: Layer, plane: str, path: Path) -> list[bool]:
    keys: list[bool] = []
    for n, p in enumerate(iter_ivf(path)):
        try:
            info = inspect_packet(bytes(p))
        except Vp9Error as exc:
            raise PackError(f"layer {L.id!r} {plane} packet {n}: not a VP9 frame ({exc})") from exc
        if info.shown_count != 1:
            raise PackError(f"layer {L.id!r} {plane} packet {n} shows {info.shown_count} frames; every packet "
                            f"must be exactly one displayed frame (alt-ref must be off)")
        if bool(p.is_keyframe) != info.key_frame:
            raise PackError(f"layer {L.id!r} {plane} packet {n}: demuxer key flag {p.is_keyframe} disagrees with "
                            f"the VP9 frame header ({info.key_frame})")
        ki = info.key_info
        if ki and (ki.width, ki.height) != L.coded:
            raise PackError(f"layer {L.id!r} {plane} packet {n} is {ki.width}x{ki.height}, expected "
                            f"{L.coded[0]}x{L.coded[1]}")
        keys.append(info.key_frame)
    if len(keys) != L.n_frames:
        raise PackError(f"layer {L.id!r} {plane}: encoder produced {len(keys)} packets, expected exactly {L.n_frames}")
    return keys


def verify_layer(proj: Project, L: Layer, log: Log) -> None:
    keys = scan_plane(L, "color", L.color_ivf)
    if L.alpha:
        akeys = scan_plane(L, "alpha", L.alpha_ivf)
        diff = [n for n, (a, b) in enumerate(zip(keys, akeys)) if a != b]
        if diff:
            raise PackError(f"layer {L.id!r}: color and alpha key frames differ at local frames {diff[:10]} (I5)")
    if not keys[0]:
        raise PackError(f"layer {L.id!r}: first frame (global {L.start_frame}) is not a key frame (I4)")
    want = [n == 0 or (L.start_frame + n) % proj.gop == 0 for n in range(L.n_frames)]
    if keys != want:
        extra = [L.start_frame + n for n in range(L.n_frames) if keys[n] and not want[n]]
        missing = [L.start_frame + n for n in range(L.n_frames) if want[n] and not keys[n]]
        log(f"  warning: layer {L.id!r} key frames deviate from the GOP grid (extra {extra[:8]}, "
            f"missing {missing[:8]}); RAPs are computed from the real ones")
    L.keyflags = keys


def compute_raps(proj: Project) -> list[bool]:
    return [all(not (L.kind == "video" and L.start_frame <= f < L.end_frame) or L.keyflags[f - L.start_frame]
                for L in proj.layers) for f in range(proj.frame_count)]


def check_rap_spacing(proj: Project, raps: list[bool]) -> None:
    if not raps[0]:
        raise PackError("frame 0 is not a RAP (I7)")
    positions = [f for f, r in enumerate(raps) if r]
    for a, b in zip(positions, positions[1:]):
        if b - a > proj.gop:
            raise PackError(
                f"RAPs at frames {a} and {b} are {b - a} frames apart, more than gop={proj.gop} (I8). Key frames of "
                f"the layers active there do not line up; align layer starts to multiples of {proj.gop} frames "
                f"({float(proj.gop / proj.fps):g} s) or re-check the encoder output.")


def build_meta(proj: Project, audio: AudioTrack | None) -> dict:
    layers, off = [], 0
    for L in proj.layers:
        if L.kind == "video":
            layers.append(M.video_layer(id=L.id, name=L.name, z=L.z, rect=L.rect, start=L.start_frame,
                                        end=L.end_frame, fps=proj.fps, alpha=L.alpha, lossless=L.lossless,
                                        blend=L.blend, opacity=L.opacity, visible=L.visible))
        else:
            layers.append(M.still_layer(id=L.id, name=L.name, z=L.z, rect=L.rect, start=L.start_frame,
                                        end=L.end_frame, offset=off, length=len(L.png), blend=L.blend,
                                        opacity=L.opacity, visible=L.visible))
            off += len(L.png)
    return M.file_meta(width=proj.width, height=proj.height, background=proj.background, fps=proj.fps,
                       frame_count=proj.frame_count, gop=proj.gop, layers=layers,
                       audio=audio.meta() if audio else None)


def write_file(proj: Project, audio: AudioTrack | None, raps: list[bool], path: Path) -> None:
    meta = build_meta(proj, audio)
    resources = b"".join(L.png for L in proj.layers if L.kind == "still")
    video = [L for L in proj.layers if L.kind == "video"]
    num, den = proj.fps.numerator, proj.fps.denominator
    path.parent.mkdir(parents=True, exist_ok=True)
    w = LVFWriter(path)
    iters = {}
    try:
        w.begin(meta, resources)
        for L in video:
            iters[(L.index, "color")] = iter_ivf(L.color_ivf)
            if L.alpha:
                iters[(L.index, "alpha")] = iter_ivf(L.alpha_ivf)
        apk = audio.packets if audio else []
        ai = 0
        for f in range(proj.frame_count):
            entries = []
            for L in video:
                if not L.start_frame <= f < L.end_frame:
                    entries.append(VideoEntry(L.index, ENTRY_EMPTY))
                    continue
                color = bytes(next(iters[(L.index, "color")]))
                alpha = bytes(next(iters[(L.index, "alpha")])) if L.alpha else b""
                key = L.keyflags[f - L.start_frame]
                entries.append(VideoEntry(L.index, ENTRY_FRAME, FRAME_FLAG_KEY if key else 0, color, alpha))
            hi = pts_us(f + 1, num, den)
            pk = []
            while ai < len(apk) and apk[ai].pts_us < hi:
                pk.append(apk[ai])
                ai += 1
            w.write_cau(Cau(f, CAU_FLAG_RAP if raps[f] else 0, entries, pk))
        w.finish()
    except BaseException:
        w.abort()
        raise
    finally:
        for it in iters.values():
            it.close()


def pack(project_path: str | Path, output: str | None = None, *, jobs: int = 4, keep_temp: bool = False,
         log: Log = print) -> Report:
    t0 = time.monotonic()
    proj = load_project(Path(project_path).resolve(), output)
    log(f"project {proj.path.name}: {proj.width}x{proj.height} @ {proj.fps} fps, {proj.frame_count} frames, "
        f"gop {proj.gop}, {len(proj.layers)} layers")
    tmp = Path(tempfile.mkdtemp(prefix="fflv_pack_"))
    try:
        jobs_list = []
        for L in proj.layers:
            log(f"  layer {L.index} {L.id!r:<12} {L.kind:<5} frames [{L.start_frame}, {L.end_frame})"
                f"{'  +alpha' if L.alpha else ''}{'  lossless' if L.lossless else ''}")
            if L.kind == "still":
                L.png = still_png(L.src)
                continue
            prepare_video_layer(L)
            L.color_ivf = tmp / f"L{L.index}_color.ivf"
            jobs_list.append((f"layer {L.id!r} color", vp9_cmd(proj, L, "color", L.color_ivf)))
            if L.alpha:
                L.alpha_ivf = tmp / f"L{L.index}_alpha.ivf"
                jobs_list.append((f"layer {L.id!r} alpha", vp9_cmd(proj, L, "alpha", L.alpha_ivf)))

        log(f"transcoding {len(jobs_list)} VP9 planes with {jobs} parallel jobs ...")
        audio: AudioTrack | None = None
        seconds = float(proj.frame_count / proj.fps)
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futs = [ex.submit(run, cmd, what) for what, cmd in jobs_list]
            afut = (ex.submit(encode_audio, proj.audio_src, max_seconds=seconds, bitrate=proj.audio_bitrate,
                              channels=proj.audio_channels) if proj.audio_src else None)
            for fu in futs:
                fu.result()
            if afut:
                audio = afut.result()
                audio.packets = audio.packets_before(pts_us(proj.frame_count, proj.fps.numerator, proj.fps.denominator))
        log(f"  done in {time.monotonic() - t0:.1f} s")

        log("verifying encoder output ...")
        for L in proj.layers:
            if L.kind == "video":
                verify_layer(proj, L, log)
        raps = compute_raps(proj)
        check_rap_spacing(proj, raps)
        log(f"  {sum(raps)} RAPs; every packet is one shown frame; color/alpha key frames match")
        if audio:
            log(f"  audio: {len(audio.packets)} Opus packets, {audio.channels} ch, pre-skip {audio.pre_skip}")
        log(f"writing {proj.output} ...")
        # Written beside the output, validated, then renamed over it: a viewer watching the output
        # (fflv view) never sees a half-written file, and a failed pack keeps the previous file.
        part = temp_path_for(proj.output)
        write_file(proj, audio, raps, part)
        rep = publish(part, proj.output)
    except MediaError as exc:
        raise PackError(str(exc)) from exc
    except InvalidOutput as exc:
        raise PackError(str(exc)) from exc
    finally:
        if keep_temp:
            log(f"  temp files kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    log(f"total {time.monotonic() - t0:.1f} s")
    return rep
