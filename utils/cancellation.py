import threading

from utils.exceptions import CancellationError


class CancellationManager:
    """A simple thread-safe class to manage cancellation state."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cancelled = False

    def cancel(self):
        """Set the cancellation flag."""
        with self._lock:
            self._cancelled = True

    def is_cancelled(self):
        """Check if cancellation has been requested."""
        with self._lock:
            return self._cancelled


# The UI runs one job at a time; this is its manager, for code that isn't handed one.
# It stays set after the run ends so abandoned background calls still see the cancel.
_active: CancellationManager | None = None


def activate(manager: CancellationManager) -> None:
    global _active
    _active = manager


def cancel_active() -> None:
    if _active is not None:
        _active.cancel()


def is_cancelled() -> bool:
    manager = _active
    return manager is not None and manager.is_cancelled()


def raise_if_cancelled() -> None:
    if is_cancelled():
        raise CancellationError("Process cancelled by user.")
