import logging
import os
import sys

from typing import Optional

_LOGGER_NAME = "project_logger"
_LOGGER = None


class RankFilter(logging.Filter):
    """Allow log records only on rank zero."""

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        return self.rank == 0


def _init_logger(log_dir: Optional[str], rank: int) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.addFilter(RankFilter(rank))

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file handler only rank0
    if log_dir is not None and rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "log.txt")

        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(log_dir: Optional[str] = None, rank: int = 0, world_size: int = 1) -> logging.Logger:
    """Return the process-global logger, creating its handlers on the first call.

    Args:
        log_dir: Optional directory for rank-zero log.txt output.
        rank: Process rank used to filter console and file output.
        world_size: Accepted for caller compatibility; no process group is created.

    Returns:
        Shared logger. Later calls reuse the original handlers and directory.
    """
    global _LOGGER

    if _LOGGER is None:
        _LOGGER = _init_logger(log_dir, rank)

    return _LOGGER
