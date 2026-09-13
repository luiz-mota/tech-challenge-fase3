"""Logger estruturado compartilhado por todos os módulos da pipeline."""

import logging
import sys


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # evita handlers duplicados quando o módulo é reimportado
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
