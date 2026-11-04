"""Logging setup: one call at startup, human-shaped in development, JSON in production.

``log_json`` exists because Render and Loki both index on structured fields, and a pipeline log full
of ``2026-11-03 12:04:11,442 INFO truelens.api: job ...`` is one grep away from useless once three
services share a stream. In development the same setting would make the terminal unreadable, so the
default is the format the reader is in.

The two things deliberately configured here rather than left alone:

* **The root logger, not our namespace only.** uvicorn, slowapi and PIL all log, and a deployment
  where ``app.*`` is JSON while ``uvicorn.error`` is colour text cannot be read by either a human or
  a log shipper.
* **Our own loggers do not propagate twice.** Attaching the handler to the root and leaving the child
  handlers in place would emit every line twice; this attaches one handler to the root and sets the
  level per namespace.

No request ever writes a filename, an upload, or any EXIF value to these logs. The only user-supplied
string that appears is the sanitized label, capped and stripped of newlines by
``orchestrator._safe_label``, and that is a deliberate exception rather than a pattern.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["configure_logging"]

_NOISY = {
    # PIL's internal decoder chatter on a malformed file is a *client* error we already log once.
    "PIL": logging.WARNING,
    "urllib3": logging.WARNING,
    # torch prints a warning per load on CPU builds; it is true, unactionable, and repeated.
    "torch": logging.WARNING,
    "asyncio": logging.WARNING,
    # One INFO line per HTTP request from the test client's transport. The app logs the events that
    # matter (job queued, job finished, rate limited); an access line per poll drowns them.
    "httpx": logging.WARNING,
}


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Install the process-wide handler. Idempotent, so tests can call it again after a settings change."""
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        from pythonjsonlogger.json import JsonFormatter

        handler.setFormatter(JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            # "level" and "time" are the field names every log UI on Render/Loki already indexes on;
            # leaving them as "levelname"/"asctime" means a dashboard that cannot sort by severity.
            rename_fields={"levelname": "level", "asctime": "time"},
        ))
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-28s %(message)s", datefmt="%H:%M:%S"))

    root = logging.getLogger()
    for existing in list(root.handlers):
        # Replacing rather than appending: a second ``configure_logging`` call in a test process
        # would otherwise print every line as many times as it has been configured.
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    for name, floor in _NOISY.items():
        logging.getLogger(name).setLevel(max(floor, logging.getLogger(name).level or 0))

    # uvicorn installs its own handlers when it boots the app programmatically. Clearing them keeps
    # its access and error lines in the same format as everything else instead of beside it.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
