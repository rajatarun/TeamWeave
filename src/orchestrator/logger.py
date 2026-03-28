import json
import logging
import os
from datetime import datetime
from typing import Any

_BASE_RECORD_KEYS = frozenset(logging.makeLogRecord({}).__dict__.keys())


class _ExtraAwareFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _BASE_RECORD_KEYS and not key.startswith("_")
        }
        if not extras:
            return message
        return f"{message} extra={jdump(extras)}"

def get_logger(name: str) -> logging.Logger:
    level = _resolve_log_level()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger
    h = logging.StreamHandler()
    h.setLevel(level)
    fmt = _ExtraAwareFormatter("%(asctime)sZ %(levelname)s %(name)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    return logger

def _resolve_log_level() -> int:
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    return getattr(logging, log_level, logging.INFO)

def jdump(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)

def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def today_ymd() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")
