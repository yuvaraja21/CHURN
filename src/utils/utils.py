import logging
from logging import FileHandler, Formatter

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    Creates and configures a logger.

    Args:
        name (str): Logger name.
        log_file (str): File to log to.
        level: Logging level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False   # stops messages from bubbling to root logger

    # Avoid adding duplicate handlers if called more than once
    for h in logger.handlers:
        # FileHandler has attribute baseFilename; compare to avoid duplicates
        if isinstance(h, FileHandler) and getattr(h, "baseFilename", None) == str(log_file):
            return logger

    handler = FileHandler(str(log_file))            # now FileHandler is defined
    handler.setFormatter(Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    return logger
