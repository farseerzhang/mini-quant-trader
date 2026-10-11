"""Logging for every command (proposal section 26).

Messages go to ``<database directory>/logs/quant.log`` (``data/logs/quant.log`` by default),
rotated by size. The log sits next to the database it describes, so a run against another
file (``--db``, or a test database) logs next to that file. Terminal output is unchanged:
the commands keep printing their reports; the log adds what happened, when, and why it failed.
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from pathlib import Path

from quant.config.settings import LoggingSettings

LOGGER_NAME = "quant"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class _UTCFormatter(logging.Formatter):
    converter = time.gmtime

    def formatTime(self, record, datefmt=None):  # noqa: N802 (logging API)
        return super().formatTime(record, datefmt or "%Y-%m-%dT%H:%M:%SZ")


def log_path(database_path: str | Path) -> Path:
    return Path(database_path).resolve().parent / "logs" / "quant.log"


def configure_logging(settings: LoggingSettings, database_path: str | Path) -> logging.Handler:
    """Send the ``quant`` loggers to the rotating log file; returns the handler.

    Calling it again (another command in the same process, e.g. in tests) replaces the
    previous file handler instead of adding a second one. ``close_logging`` detaches it.
    """
    path = log_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, "_quant_file", False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=int(settings.max_megabytes * 1024 * 1024), backupCount=settings.backups, encoding="utf-8"
    )
    handler._quant_file = True
    handler.setFormatter(_UTCFormatter(FORMAT))
    logger.addHandler(handler)
    logger.setLevel(settings.level)
    logger.propagate = False
    return handler


def close_logging(handler: logging.Handler) -> None:
    """Detach and close the file handler (the file is not held open between commands)."""
    logging.getLogger(LOGGER_NAME).removeHandler(handler)
    handler.close()
