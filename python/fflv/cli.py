"""The `fflv` command (`fflv --help`); implemented by the Rust core."""

from __future__ import annotations

import signal
import sys

from . import _fflv


def main(argv: list[str] | None = None) -> int:
    """Run the command line with `argv` (default: sys.argv[1:]); returns the exit code."""
    args = sys.argv[1:] if argv is None else list(argv)
    sys.stdout.flush()
    return _fflv.cli_main(["fflv", *map(str, args)])


def entry() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C ends `fflv view` like any other command line tool
    sys.exit(main())
