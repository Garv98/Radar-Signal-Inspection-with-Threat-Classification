"""
Production-Grade Logging Utilities

Provides structured logging with file and console handlers,
performance metrics tracking, and audit logging for DRDO compliance.
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import traceback
from functools import wraps
import time


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data['data'] = record.extra_data

        # Add exception info
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info) if record.exc_info[0] else None
            }

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for human readability."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_dir: str = 'outputs/logs',
    level: int = logging.INFO,
    structured: bool = True,
    console: bool = True,
    log_name: Optional[str] = None
) -> logging.Logger:
    """
    Set up production logging configuration.

    Args:
        log_dir: Directory for log files
        level: Logging level
        structured: Use JSON structured logging for files
        console: Enable console output
        log_name: Custom log file name (default: timestamp)

    Returns:
        Configured root logger
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # File handler with structured JSON
    if log_name is None:
        log_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_file = Path(log_dir) / f'{log_name}.log'

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    if structured:
        file_handler.setFormatter(StructuredFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
        ))
    logger.addHandler(file_handler)

    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


class PerformanceMonitor:
    """
    Performance monitoring for inference latency and throughput.
    Critical for DRDO real-time requirements.
    """

    def __init__(self, name: str = 'default'):
        self.name = name
        self.timings = []
        self.logger = logging.getLogger(f'perf.{name}')

    def record(self, latency_ms: float, batch_size: int = 1):
        """Record a timing measurement."""
        self.timings.append({
            'timestamp': datetime.utcnow().isoformat(),
            'latency_ms': latency_ms,
            'batch_size': batch_size,
            'throughput': batch_size / (latency_ms / 1000) if latency_ms > 0 else 0
        })

    def get_statistics(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.timings:
            return {}

        latencies = [t['latency_ms'] for t in self.timings]
        throughputs = [t['throughput'] for t in self.timings]

        import numpy as np
        return {
            'count': len(self.timings),
            'mean_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies),
            'mean_throughput': np.mean(throughputs),
            'total_samples': sum(t['batch_size'] for t in self.timings)
        }

    def check_sla(self, max_latency_ms: float = 50.0) -> bool:
        """Check if performance meets SLA requirements."""
        stats = self.get_statistics()
        if not stats:
            return True

        meets_sla = stats['p95_latency_ms'] <= max_latency_ms
        if not meets_sla:
            self.logger.warning(
                f"SLA VIOLATION: p95 latency {stats['p95_latency_ms']:.2f}ms > {max_latency_ms}ms"
            )
        return meets_sla

    def log_report(self):
        """Log performance report."""
        stats = self.get_statistics()
        if stats:
            self.logger.info(f"Performance Report for '{self.name}':")
            for key, value in stats.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.2f}")
                else:
                    self.logger.info(f"  {key}: {value}")


def timed(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} executed in {elapsed_ms:.2f}ms")

        return result
    return wrapper


class AuditLogger:
    """
    Audit logging for compliance and traceability.
    Records all model predictions with timestamps and confidence.
    """

    def __init__(self, log_file: str = 'outputs/logs/audit.jsonl'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('audit')

    def log_prediction(
        self,
        input_id: str,
        prediction: str,
        confidence: float,
        threat_level: float,
        latency_ms: float,
        metadata: Optional[Dict] = None
    ):
        """Log a prediction for audit trail."""
        record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'input_id': input_id,
            'prediction': prediction,
            'confidence': confidence,
            'threat_level': threat_level,
            'latency_ms': latency_ms,
            'metadata': metadata or {}
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

        self.logger.info(
            f"PREDICTION: {input_id} -> {prediction} "
            f"(conf={confidence:.3f}, threat={threat_level:.2f}, latency={latency_ms:.1f}ms)"
        )

    def get_recent_predictions(self, n: int = 100) -> list:
        """Get recent predictions from audit log."""
        if not self.log_file.exists():
            return []

        predictions = []
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))

        return predictions[-n:]
