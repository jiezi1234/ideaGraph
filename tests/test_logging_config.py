import logging
from datetime import date

from src.logging_config import configure_logging


def test_configure_logging_adds_console_and_file_handlers(tmp_path):
    log_file = tmp_path / "ideagraph.log"

    configure_logging(level="INFO", log_file=log_file)

    root_logger = logging.getLogger()
    handler_types = {type(handler).__name__ for handler in root_logger.handlers}

    assert "StreamHandler" in handler_types
    assert "FileHandler" in handler_types


def test_configure_logging_writes_to_dated_log_file(tmp_path):
    log_file = tmp_path / "ideagraph.log"

    configure_logging(level="INFO", log_file=log_file)

    dated_log_file = tmp_path / f"ideagraph-{date.today().isoformat()}.log"

    assert dated_log_file.exists()
