"""Logging utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    console: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Configure root logger."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class TrainingLogger:
    """Structured training logger with step tracking."""
    
    def __init__(self, name: str, log_dir: Optional[str | Path] = None):
        self.logger = get_logger(name)
        self.step = 0
        self.epoch = 0
        self.log_dir = Path(log_dir) if log_dir else None
        self._metrics_file = None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._metrics_file = open(self.log_dir / "metrics.jsonl", "a")
    
    def log(self, level: str, message: str, **kwargs) -> None:
        """Log a message with optional structured data."""
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        full_message = f"{message} {extra}" if extra else message
        getattr(self.logger, level.lower())(full_message)
    
    def info(self, message: str, **kwargs) -> None:
        self.log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self.log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        self.log("ERROR", message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        self.log("DEBUG", message, **kwargs)
    
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log training metrics."""
        if step is not None:
            self.step = step
        # Remove 'step' from metrics to avoid conflict
        metrics_to_log = {k: v for k, v in metrics.items() if k != "step"}
        self.info("Metrics", step=self.step, **metrics_to_log)
        
        if self._metrics_file:
            import json
            record = {"step": self.step, **metrics}
            self._metrics_file.write(json.dumps(record) + "\n")
            self._metrics_file.flush()
    
    def set_step(self, step: int) -> None:
        self.step = step
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
    
    def close(self) -> None:
        if self._metrics_file:
            self._metrics_file.close()
            self._metrics_file = None