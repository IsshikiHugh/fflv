"""Validation reports (see `fflv.validate`)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

INVARIANTS = tuple(f"I{i}" for i in range(1, 11))


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    frame: int | None
    severity: str  # "error" | "warning"

    def __str__(self) -> str:
        where = f" frame {self.frame}:" if self.frame is not None else ""
        return f"[{self.code}]{where} {self.message}"


@dataclass(frozen=True)
class LayerStats:
    frames: int
    keyframes: int
    color_bytes: int
    alpha_bytes: int


@dataclass
class Report:
    """What the validator found. Invariant violations use the invariant's own code ("I1" …
    "I10"); other codes: HDR, META, RES, CAU, VP9, AUD."""

    path: str
    issues: list[Issue] = field(default_factory=list)
    suppressed: dict[str, int] = field(default_factory=dict)  # "CODE/severity" → issues not recorded
    fatal: bool = False
    meta: dict | None = None
    cau_count: int = 0
    rap_frames: list[int] = field(default_factory=list)
    layer_stats: dict[int, LayerStats] = field(default_factory=dict)
    audio_packets: int = 0
    audio_bytes: int = 0
    file_size: int = 0

    @classmethod
    def from_json(cls, text: str | None) -> "Report | None":
        if text is None:
            return None
        d = json.loads(text)
        d["issues"] = [Issue(**i) for i in d["issues"]]
        d["layer_stats"] = {int(k): LayerStats(**v) for k, v in d["layer_stats"].items()}
        return cls(**d)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def error_codes(self) -> set[str]:
        codes = {i.code for i in self.errors}
        codes |= {k[: -len("/error")] for k in self.suppressed if k.endswith("/error")}
        return codes

    @property
    def error_count(self) -> int:
        return len(self.errors) + sum(n for k, n in self.suppressed.items() if k.endswith("/error"))

    @property
    def ok(self) -> bool:
        return not self.fatal and not self.error_codes()

    def __repr__(self) -> str:
        state = "valid" if self.ok else f"INVALID, {self.error_count} errors ({', '.join(sorted(self.error_codes()))})"
        return f"<fflv.Report {self.path}: {state}, {self.cau_count} frames>"
