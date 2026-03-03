import sys
from loguru import logger
from config import config

def setup_logger():
    """Logger setup"""

    logger.remove()
    logger.add(
        sys.stdout,
        level=config["logging"]["level"],
        colorize=True,
    )

    return logger

def get_logger(name: str):
    """Get logger"""

    return logger.bind(name=name)