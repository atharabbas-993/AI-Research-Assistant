# src/logger.py

import logging
import os

from src.config import LOG_LEVEL, LOG_FILE


def setup_logger(name: str) -> logging.Logger:
    """
    Creates a configured logger that writes to both the console
    and a log file. Call this once per module, using __name__
    so log messages show which file they came from.

    Args:
        name (str): Usually the calling module's __name__.

    Returns:
        logging.Logger: A ready-to-use logger instance.
    """

    # Make sure the logs folder exists before we try to write to it
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Avoid adding duplicate handlers if this function is called
    # multiple times for the same logger (can happen with --reload)
    if not logger.handlers:
        # Formatter defines what each log line looks like:
        # timestamp - module name - level - message
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Handler 1: print logs to the console (useful during development)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler 2: also save logs to a file (useful for later review,
        # and required once this runs on a server with no visible console)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger