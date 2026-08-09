"""Structured logging setup, called once from api.py's lifespan before
anything else logs.

Emits one JSON line per record with a "severity" key -- Cloud Logging's
structured-logging convention for stdout/stderr. Without it, Cloud Run
ingests every line as INFO regardless of actual level, which defeats the
point of having levels at all once this is actually deployed there.
"""

import json
import logging


class _CloudLoggingFormatter(logging.Formatter):
    _LEVEL_TO_SEVERITY = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": self._LEVEL_TO_SEVERITY.get(record.levelno, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_CloudLoggingFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # These libraries log a lot at INFO/DEBUG (HTTP client internals, model
    # loading progress bars re-logged as lines, etc.) that's noise here even
    # when our own code is at DEBUG -- cap them at WARNING regardless.
    for noisy_logger in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
