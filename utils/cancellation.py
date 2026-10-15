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
_active: CancellationManager | None = None
# Abandoned background calls keep their run's manager after the next run activates one
_pinned = threading.local()


def activate(manager: CancellationManager) -> None:
    global _active
    _active = manager


def active_manager() -> CancellationManager | None:
    return _active


def pin_thread(manager: CancellationManager | None) -> None:
    """Make this thread check `manager` instead of whichever one is active later."""
    _pinned.manager = manager


def cancel_active() -> None:
    if _active is not None:
        _active.cancel()


def is_cancelled() -> bool:
    manager = getattr(_pinned, "manager", None) or _active
    return manager is not None and manager.is_cancelled()


def raise_if_cancelled() -> None:
    if is_cancelled():
        raise CancellationError("Process cancelled by user.")
