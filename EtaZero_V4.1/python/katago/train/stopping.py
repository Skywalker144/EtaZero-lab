"""Defer single-device learner stop signals to a complete batch boundary."""
from contextlib import contextmanager
import signal

_requested = False
_active = False


def requested():
    return _requested


def _request(*_):
    global _requested
    _requested = True


@contextmanager
def defer_signals():
    # timed_learner also covers imports and compiler receipt publication. The
    # standalone learner uses the same handler without resetting a pending stop.
    global _requested, _active
    if _active:
        yield
        return
    _requested, _active = False, True
    handlers = {sig: signal.signal(sig, _request) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        _active = False
