"""Persistent native GTP transport for the outer workbench.

Game state, move choice and analysis remain in KataGo. This module only frames
commands, delivers source JSON reports and interrupts the native search.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import select
import subprocess
import tempfile
from threading import Lock, RLock
import time

from . import ROOT


class GtpError(RuntimeError):
    """The native engine rejected a command; its process remains usable."""


@dataclass
class Reply:
    body: str
    analysis: dict | None = None


class GtpProcess:
    def __init__(self, config, model, *, binary=None, human_model=None, overrides='', timeout=120):
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('GTP timeout must be finite and positive')
        self.timeout = timeout
        self.process = None
        self.errors = tempfile.TemporaryFile(mode='w+b')
        self.buffer = bytearray()
        self.command_lock = Lock()
        self.write_lock = RLock()
        self.sequence = 0
        self.active_id = None
        config = Path(config).resolve()
        self.argv = [str(Path(binary or ROOT / 'build/katago').resolve()), 'gtp',
                     '-config', str(config), '-model', str(Path(model).resolve())]
        if human_model is not None:
            self.argv += ['-human-model', str(Path(human_model).resolve())]
        # Native config is consumed in its own directory, including relative
        # resource paths. Only stdout logging is disabled for GTP framing.
        self.argv += ['-override-config', ','.join(filter(None, [overrides, 'logToStdout=false']))]
        try:
            self.process = subprocess.Popen(self.argv, cwd=config.parent, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.errors, bufsize=0)
            if self.command('protocol_version').body != '2':
                raise RuntimeError('The native engine did not negotiate GTP 2')
            self.models = json.loads(self.command('kata-get-models').body)
            # Also verify the structured state boundary before publishing a
            # workbench session; an older binary must not look connected.
            self.state()
        except BaseException:
            self.close()
            raise

    def _diagnostic(self):
        self.errors.seek(0, os.SEEK_END)
        self.errors.seek(max(0, self.errors.tell() - 8000))
        return self.errors.read().decode('utf-8', errors='replace').strip()

    def _line(self, unlimited_wait=False):
        deadline = None if unlimited_wait else time.monotonic() + self.timeout
        while b'\n' not in self.buffer:
            remaining = None if deadline is None else deadline - time.monotonic()
            if (remaining is not None and remaining <= 0) or not select.select([self.process.stdout], [], [], remaining)[0]:
                self.close()
                raise RuntimeError('Native GTP response timed out; create a new session')
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                detail = self._diagnostic()
                self.close()
                raise RuntimeError(detail or 'Native GTP process exited; create a new session')
            self.buffer.extend(block)
        line, _, self.buffer = self.buffer.partition(b'\n')
        return line.decode('utf-8').rstrip('\r')

    def command(self, command, *, on_report=None, should_stop=None, unlimited_wait=False):
        if not isinstance(command, str) or not command.strip() or any(ord(c) < 32 or ord(c) == 127 for c in command):
            raise ValueError('A GTP command must be a single nonempty line')
        if command.lstrip()[0].isdigit():
            raise ValueError('GTP request IDs belong to the transport')
        with self.command_lock:
            with self.write_lock:
                if self.process is None or self.process.poll() is not None:
                    raise RuntimeError('Native GTP process is not running; create a new session')
                self.sequence += 1
                request_id = self.sequence
                self.active_id = request_id
                try:
                    self.process.stdin.write(f'{request_id} {command}\n'.encode())
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError) as error:
                    self.active_id = None
                    raise RuntimeError('Native GTP process disconnected') from error
            try:
                if should_stop is not None and should_stop():
                    self.stop()
                line = self._line(unlimited_wait)
                header = re.fullmatch(r'([=?])(\d+)(?: (.*))?', line)
                if header is None or int(header[2]) != request_id:
                    self.close()
                    raise RuntimeError(f'Unexpected native GTP response header: {line[:160]}')
                body = [header[3]] if header[3] else []
                analysis = None
                while (line := self._line(unlimited_wait)) != '':
                    if on_report is not None and line.startswith('{') and header[1] == '=':
                        analysis = json.loads(line)
                        on_report(analysis)
                    else:
                        body.append(line)
                text = '\n'.join(body)
            except BaseException:
                # A callback/decoder failure leaves unread framing in the pipe.
                # Terminate this process rather than reusing an ambiguous stream.
                self.close()
                raise
            finally:
                with self.write_lock:
                    self.active_id = None
            if header[1] == '?':
                raise GtpError(text)
            return Reply(text, analysis)

    def state(self):
        return json.loads(self.command('kata-get-state').body)

    def analysis(self):
        return json.loads(self.command('kata-get-analysis').body)

    def stop(self):
        """Interrupt a cancellable search/analysis using the source GTP newline."""
        with self.write_lock:
            if self.active_id is not None and self.process is not None and self.process.poll() is None:
                try:
                    self.process.stdin.write(b'\n')
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass  # The command reader reports the actual process failure.

    def close(self):
        process = self.process
        if process is not None:
            with self.write_lock:
                if process.stdin is not None and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
        self.errors.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
