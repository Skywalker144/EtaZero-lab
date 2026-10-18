import json
from pathlib import Path
import select
import os
import shlex
import time
from threading import Lock, RLock
import subprocess
import tempfile
from typing import Any

from etazero.config import write_native
from etazero.analysis import model_info


class Engine:
    def __init__(self, binary: Path, model: Path, config: dict[str, Any], timeout: float = 120):
        self.timeout = timeout
        self.directory = tempfile.TemporaryDirectory(prefix='etazero-session-')
        self.errors = tempfile.TemporaryFile(mode='w+t')
        self.process = None
        self.buffer = bytearray()
        self.write_lock = RLock()
        self.command_lock = Lock()
        self.sequence = 0
        self.active_id = None
        try:
            resolved = Path(self.directory.name) / 'analysis.cfg'
            model, info = model_info(model)
            write_native({**config, 'network': {'canvas': info['canvas']}}, resolved)
            analysis = config['analysis']
            self.process = subprocess.Popen(
                [str(binary.resolve()), 'analysis', '--stream', 'true', '--protocol', 'json', '--config', str(resolved), '--model', str(model),
                 '--model-id', info['id'], '--device', analysis['device'], '--seed', str(analysis['seed'])],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors,
                bufsize=0,
            )
            self.canvas = self.receive()['canvas_size']
        except BaseException:
            self.close()
            raise

    def receive(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                self.close()
                raise RuntimeError('模型响应超时，请重新开始一局')
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                self.process.wait(timeout=5)
                self.errors.seek(0)
                raise RuntimeError(self.errors.read()[-4000:].strip() or '模型进程已退出，请重新开始一局')
            self.buffer.extend(block)
        line, _, self.buffer = self.buffer.partition(b'\n')
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            self.close()
            raise RuntimeError('模型返回了无效响应') from error

    def send(self, data):
        with self.write_lock:
            if self.process.poll() is not None:
                raise RuntimeError('模型进程已退出，请重新开始一局')
            self.sequence += 1
            request = {**data, 'id': self.sequence}
            try:
                self.process.stdin.write((json.dumps(request) + '\n').encode())
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise RuntimeError('模型进程已断开，请重新开始一局') from error
            return self.sequence

    def command(self, command, on_report=None, should_stop=None) -> dict[str, Any]:
        # Retain the small Python convenience API; the transport itself is JSON.
        try:
            if isinstance(command, str):
                parts = shlex.split(command)
                data = {'command': parts[0]}
                if parts[0] == 'new':
                    data.update(size=int(parts[1]), rule=parts[2])
                    if len(parts) == 6:
                        data.update(opening=parts[3], seed=parts[4], timeout_ms=int(parts[5]))
                elif parts[0] == 'play': data['action'] = int(parts[1])
                elif parts[0] == 'undo': data['count'] = int(parts[1])
                elif parts[0] in ('analyze', 'genmove'): data['visits'] = int(parts[1])
            else:
                data = dict(command)
        except (ValueError, IndexError) as error:
            raise RuntimeError("Invalid engine command arguments") from error
        data.setdefault('report_interval_ms', 250 if on_report else 0)
        with self.command_lock:
            # Publish the id under the write lock before cancellation may inspect it.
            with self.write_lock:
                request_id = self.send(data)
                self.active_id = request_id
            if should_stop and should_stop():
                self.stop()
            try:
                while True:
                    response = self.receive()
                    if response.get('id') != request_id:
                        continue  # A stop acknowledgment may arrive between commands.
                    if not response.get('ok'):
                        raise RuntimeError(response.get('error', '模型请求失败'))
                    if response.get('final'):
                        return response
                    if on_report:
                        on_report(response['analysis'])
            finally:
                with self.write_lock:
                    self.active_id = None

    def stop(self):
        with self.write_lock:
            target = self.active_id
        if target is not None:
            self.send({'command': 'stop', 'target_id': target})

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        self.errors.close()
        self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
