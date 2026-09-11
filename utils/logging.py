import atexit
import threading
import time
from pathlib import Path
from typing import TextIO

CURRENT_LOG_NAME = "mangatranslator.log"
PREVIOUS_LOG_NAME = "mangatranslator.prev.log"

_log_lock = threading.Lock()
_log_file: TextIO | None = None
_log_initialized = False


def init_file_logging(
    log_dir: Path | str | None = None, force_reinit: bool = False
) -> None:
    """Initialize file logging in log_dir with rotation (current and one previous)."""
    global _log_file, _log_initialized

    with _log_lock:
        if _log_initialized and not force_reinit:
            return

        if _log_file is not None and not _log_file.closed:
            _log_file.close()
            _log_file = None

        if log_dir is None:
            target_dir = Path(__file__).resolve().parent.parent
        else:
            target_dir = Path(log_dir)

        target_dir.mkdir(parents=True, exist_ok=True)
        curr_log = target_dir / CURRENT_LOG_NAME
        prev_log = target_dir / PREVIOUS_LOG_NAME

        if curr_log.exists():
            if prev_log.exists():
                prev_log.unlink()
            curr_log.replace(prev_log)

        _log_file = open(curr_log, "a", encoding="utf-8")  # noqa: SIM115
        _log_initialized = True


def close_file_logging() -> None:
    """Close the current log file and reset initialization state."""
    global _log_file, _log_initialized

    with _log_lock:
        if _log_file is not None and not _log_file.closed:
            _log_file.close()
        _log_file = None
        _log_initialized = False


atexit.register(close_file_logging)


def log_message(message, verbose=False, always_print=False):
    """
    Log a message to file and optionally print to console.

    File logging always records all messages (including verbose).
    Console output is filtered based on verbose or always_print.

    Args:
        message (str): The message to print
        verbose (bool): Whether to print detailed logs
        always_print (bool): Whether to print regardless of verbose setting
    """
    if not _log_initialized:
        init_file_logging()

    with _log_lock:
        if _log_file is not None and not _log_file.closed:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            level = "INFO" if always_print else "VERBOSE"
            _log_file.write(f"[{timestamp}] [{level}] {message}\n")
            _log_file.flush()

        if verbose or always_print:
            print(f"{message}")
