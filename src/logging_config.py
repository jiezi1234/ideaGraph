from __future__ import annotations

import logging
from datetime import date
from pathlib import Path


def _dated_log_file(log_file: str | Path) -> Path:
    resolved_log_file = Path(log_file)
    return resolved_log_file.with_name(
        f"{resolved_log_file.stem}-{date.today().isoformat()}{resolved_log_file.suffix}"
    )


def configure_logging(level: str = "INFO", log_file: str | Path = "logs/ideagraph.log") -> None:
    resolved_log_file = _dated_log_file(log_file)
    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
