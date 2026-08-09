"""Console setup helpers.

The SentencePiece vocabulary contains U+2581 (LOWER ONE EIGHTH BLOCK, the word
boundary marker), which the default Windows console codepage (cp1252) cannot
encode. Printing a token list therefore raises ``UnicodeEncodeError`` rather
than producing mangled output, which turns a debug print into a crash.
"""

from __future__ import annotations

import logging
import sys


def configure_stdout() -> None:
    """Force UTF-8 on stdout/stderr, replacing anything unencodable."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Redirected to a pipe that refuses reconfiguration; the
                # replace-on-error behaviour below still applies.
                pass


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Set up log formatting for CLI entry points."""
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
