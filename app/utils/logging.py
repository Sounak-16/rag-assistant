import logging
import re
import sys
from typing import Any


class SecretMaskingFormatter(logging.Formatter):
    """Formatter that strips and masks potential API keys and secrets from log messages."""

    PATTERNS = [
        re.compile(r"(api[_-]?key|secret|token|password|bearer\s+)[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?", re.IGNORECASE),
        re.compile(r"(AIza[0-9A-Za-z-_]{35})"),  # Google API key
        re.compile(r"(sk-[a-zA-Z0-9]{20,})"),     # OpenAI API key
    ]

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        sanitized = original
        for pattern in self.PATTERNS:
            sanitized = pattern.sub(r"\1: [REDACTED]", sanitized)
        return sanitized


def setup_logger(name: str = "rag_assistant", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if setup_logger is called repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        formatter = SecretMaskingFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
