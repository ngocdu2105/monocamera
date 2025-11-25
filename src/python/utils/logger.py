import logging
import os
from datetime import datetime


def setup_logger(name: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_filename = datetime.now().strftime("logs/%Y-%m-%d.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_filename)
    ch = logging.StreamHandler()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
logger = setup_logger("app")