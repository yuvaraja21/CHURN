# tests/test_logger.py
import logging
import re
from pathlib import Path

# import the real function (you found it in src/utils/utils.py)
from src.utils.utils import setup_logger

def test_setup_logger_creates_file_and_writes(tmp_path):
    log_file = tmp_path / "test.log"
    logger_name = "tests.mylogger"

    # create logger and write a message
    logger = setup_logger(logger_name, str(log_file), level=logging.INFO)
    logger.info("hello world")

    # flush handlers to be safe
    for h in logger.handlers:
        if hasattr(h, "flush"):
            h.flush()

    # file should exist and contain the message
    assert log_file.exists(), "Log file was not created"
    text = log_file.read_text()
    assert "hello world" in text
    assert logger_name in text
    assert "INFO" in text

def test_logger_returns_same_named_logger_and_level(tmp_path):
    log_file = tmp_path / "same.log"
    name = "tests.same_logger"

    logger1 = setup_logger(name, str(log_file), level=logging.WARNING)
    logger2 = logging.getLogger(name)  # should be same logger object
    assert logger1 is logger2
    assert logger1.level == logging.WARNING

def test_handler_is_filehandler_with_formatter(tmp_path):
    log_file = tmp_path / "fmt.log"
    name = "tests.fmt"

    logger = setup_logger(name, str(log_file))
    # ensure at least one FileHandler attached
    handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert handlers, "No FileHandler attached to the logger"
    fh = handlers[0]

    # write a message and check formatted output
    logger.info("format-test")
    for h in logger.handlers:
        if hasattr(h, "flush"):
            h.flush()
    content = log_file.read_text()

    # quick check for timestamp + name + level + message
    assert name in content and "INFO" in content and "format-test" in content
