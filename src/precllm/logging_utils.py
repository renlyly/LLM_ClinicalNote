from __future__ import annotations

import logging


def configure_logging(level: str) -> None:
    """Configure package-level logging with a predictable enterprise format."""

    resolved_level = level.upper()
    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
