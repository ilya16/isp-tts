""" Console and File Logger. """
from __future__ import annotations

from loguru import logger


def setup_logger(file: str | None = None, is_main_process: bool = True, **extra):
    import sys

    time = "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
    level = "<level>{level:<7}</level>"
    message = "<level>{message}</level>"

    formatter = f"{time} {level} - {message}"
    handlers = [dict(sink=sys.stdout, format=formatter, enqueue=True)]
    if file is not None:
        handlers.append(dict(sink=file, format=formatter, enqueue=True))

    logger.configure(
        handlers=handlers if is_main_process else [],
        extra=extra
    )

    return logger
