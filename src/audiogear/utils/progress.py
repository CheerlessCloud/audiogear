"""TTY-aware tqdm wrapper (BUG-8).

tqdm writes one progress line per redraw. Attached to a terminal that's a single
rewritten line, but when stderr is redirected to a *file log* (the usual run
setup) every redraw becomes a new line and the log balloons to tens of MB,
burying the run markers.

This drop-in wrapper:
  - disables the bar when stderr is not a TTY (file/pipe), and
  - throttles redraws (``mininterval``) when it is enabled,

both overridable per call and via the ``AUDIOGEAR_PROGRESS`` env var
(``0``/``1`` to force off/on). Import ``tqdm`` from here instead of from the
``tqdm`` package to get this behaviour everywhere.
"""

from __future__ import annotations

import os
import sys

from tqdm import tqdm as _tqdm


def _progress_disabled() -> bool:
    env = os.environ.get("AUDIOGEAR_PROGRESS")
    if env is not None and env.strip() != "":
        return env.strip().lower() in ("0", "false", "no", "off")
    try:
        return not sys.stderr.isatty()
    except Exception:  # pragma: no cover - exotic stderr
        return True


def tqdm(*args, **kwargs):
    """``tqdm`` preconfigured to stay quiet in non-interactive runs."""
    kwargs.setdefault("disable", _progress_disabled())
    kwargs.setdefault("mininterval", 1.0)  # redraw at most ~1/s when enabled
    return _tqdm(*args, **kwargs)
