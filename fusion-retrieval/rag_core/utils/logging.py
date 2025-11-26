"""Logging utilities"""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger with a sensible default format.

    `basicConfig` is only applied once, so repeated calls are safe.
    """
    logger = logging.getLogger(name or "fusion_rag")
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    return logger

