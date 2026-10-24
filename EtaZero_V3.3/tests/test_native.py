"""Log failures must not change the persistent worker request protocol."""
import io
from pathlib import Path
import queue
import sys
import time
from types import SimpleNamespace
import warnings

import pytest

from etazero.run.native import NativeWorker


WORKER = '''
import json, sys
stream = sys.stdin.buffer
for command in iter(stream.readline, b''):
    if command == b'release\\n':
        print(json.dumps({'event': 'worker_released'}), flush=True)
        continue
    fields = []
    for _ in range(8):
        size = int(stream.readline())
        fields.append(stream.read(size).decode())
        assert stream.read(1) == b'\\n'
    print(json.dumps({'event': 'worker_complete', 'attempt_id': fields[4], 'code': 0}), flush=True)
'''


@pytest.mark.parametrize('failure', ['open', 'write', 'close', 'stderr'])
def test_log_failure_allows_completion_and_next_request(tmp_path, monkeypatch, failure):
    original = Path.open
    class BrokenLog(io.StringIO):
        def write(self, value):
            if failure == 'write':
                raise OSError('injected log write')
            return super().write(value)

        def close(self):
            super().close()
            if failure == 'close':
                raise OSError('injected log close')

    def open_log(path, *args, **kwargs):
        if path == tmp_path/'stdout':
            if failure == 'open':
                raise OSError('injected log open')
            if failure in ('write', 'close'):
                return BrokenLog()
        if path == tmp_path/'stderr' and failure == 'stderr':
            raise OSError('injected stderr open')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_log)
    with warnings.catch_warnings(record=True) as reported:
        warnings.simplefilter('always')
        worker = NativeWorker([sys.executable, '-u', '-c', WORKER], tmp_path/'stderr')
        try:
            for attempt in ('first', 'second'):
                fields = ['', 'random', 1, str(tmp_path), attempt, 0, 1, 'random']
                worker.submit(fields, tmp_path/'stdout')
                with pytest.raises(RuntimeError, match='outstanding request'):
                    worker.submit(fields, tmp_path/'stdout')
                deadline = time.monotonic() + 5
                while (result := worker.take()) is None and time.monotonic() < deadline:
                    time.sleep(.005)
                assert result == {'event': 'worker_complete', 'attempt_id': attempt, 'code': 0}
                assert worker.failure is None
                worker.release()
        finally:
            worker.close()
    assert any('injected' in str(w.message) for w in reported)


@pytest.mark.parametrize('response', [b'not json\n', b'{"event":"worker_complete","attempt_id":"other"}\n'])
def test_protocol_errors_remain_fatal_without_a_log(response):
    worker = NativeWorker.__new__(NativeWorker)
    worker.process = SimpleNamespace(stdout=io.BytesIO(response))
    worker.responses = queue.Queue(maxsize=2)
    worker.attempt = 'expected'
    worker.log = worker.failure = None
    worker.closing = True
    worker.stderr_path = Path('unused.stderr')
    worker._read()
    assert isinstance(worker.failure, (ValueError, RuntimeError))
    with pytest.raises(RuntimeError, match='Native worker failed'):
        worker.take()
