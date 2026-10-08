from __future__ import annotations

import json
import logging
import sys
from typing import Any


LOGGER_NAME = "deepresearch"


def configure_logging(*, quiet: bool = False) -> None:
    """Configure concise terminal logs for CLI runs."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.WARNING if quiet else logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)


def log_step(agent: str, event: str, **fields: Any) -> None:
    """Emit one machine-searchable, human-readable step without secrets."""
    details = " ".join(
        f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
        for key, value in fields.items()
        if value is not None
    )
    message = f"[{agent}] {event}"
    if details:
        message = f"{message} {details}"
    logging.getLogger(LOGGER_NAME).info(message)
