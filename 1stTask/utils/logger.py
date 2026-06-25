"""
utils/logger.py
---------------
Provides a pre-configured logger for the entire project.
Import and call `get_logger(__name__)` in every module.
"""

import logging
import io
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that writes INFO+ to stdout with a timestamped format.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers on repeated imports
        return logger

    logger.setLevel(logging.DEBUG)

    # Force UTF-8 on Windows terminals that default to cp1252
    stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "buffer") else sys.stdout
    handler = logging.StreamHandler(stdout_utf8)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
