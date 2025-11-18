import threading


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
