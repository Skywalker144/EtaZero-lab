"""Stop owned stages without interrupting their publication or cleanup."""
import os
import signal


class StopRequested(KeyboardInterrupt):
    """The controller observed the request and waited for its child to exit."""

    def __init__(self, started=True):
        self.started = started


class StopRequest:
    def __init__(self):
        self.requested = False
        self.child = None

    def __enter__(self):
        self.handlers = {sig: signal.signal(sig, self.request)
                         for sig in (signal.SIGINT, signal.SIGTERM)}
        return self

    def __exit__(self, *_):
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)

    def request(self, *_):
        self.requested = True
        self.notify_child()

    def notify_child(self):
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def check(self, *, started=True):
        if self.requested:
            raise StopRequested(started)
