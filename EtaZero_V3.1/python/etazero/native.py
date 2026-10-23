"""Owned, persistent native selfplay process with one bounded request at a time."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import warnings


class NativeWorker:
    def __init__(self, command, stderr):
        self.command=command;self.stderr_path=Path(stderr)
        self.responses=queue.Queue(maxsize=2)
        self.log=None;self.attempt=None;self.failure=None;self.closing=False
        self.stderr=None
        try:
            self.stderr=self.stderr_path.open('wb')
        except OSError as error:
            warnings.warn(f'Native stderr log unavailable; using inherited stderr: {error}', RuntimeWarning)
        env=dict(os.environ)
        env['ETAZERO_NAN_DIAGNOSTIC_DIR']=str(self.stderr_path.parent/'nan_diagnostics')
        try:
            self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                          stderr=self.stderr,start_new_session=True,env=env)
        except BaseException:
            if self.stderr is not None:
                self.stderr.close()
            raise
        self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start()

    def _close_log(self):
        log,self.log=self.log,None
        if log is not None:
            try:
                log.close()
            except OSError as error:
                warnings.warn(f'Native log close failed: {error}', RuntimeWarning)

    def _read(self):
        try:
            for data in self.process.stdout:
                line=data.decode('utf-8')
                if self.log is not None:
                    try:
                        self.log.write(line)
                    except OSError as error:
                        warnings.warn(f'Native log write failed; protocol continues: {error}', RuntimeWarning)
                        self._close_log()
                record=json.loads(line)
                if record['event']=='worker_complete':
                    if self.attempt is None or record['attempt_id']!=self.attempt:
                        raise RuntimeError('Native worker returned a different attempt')
                    self._close_log();self.attempt=None
                    self.responses.put_nowait(record)
                elif record['event']=='worker_released':
                    self.responses.put_nowait(record)
            if not self.closing:
                raise RuntimeError('Native worker exited; inspect '+str(self.stderr_path))
        except BaseException as error:
            self.failure=error
        finally:
            self._close_log()

    def submit(self, fields, stdout):
        if self.failure:
            raise RuntimeError('Native worker reader failed') from self.failure
        if self.attempt is not None or not self.responses.empty():
            raise RuntimeError('Native worker already has an outstanding request')
        self.attempt=fields[4]
        try:
            self.log=Path(stdout).open('w')
        except OSError as error:
            warnings.warn(f'Native log unavailable; protocol continues: {error}', RuntimeWarning)
        payload=bytearray(b'selfplay\n')
        # Byte lengths permit whitespace, Unicode and newlines in filesystem paths.
        for field in fields:
            value=str(field).encode('utf-8')
            payload.extend(str(len(value)).encode()+b'\n'+value+b'\n')
        try:
            self.process.stdin.write(payload);self.process.stdin.flush()
        except BaseException:
            self._close_log()
            raise

    def take(self):
        try:
            return self.responses.get_nowait()
        except queue.Empty:
            if self.failure:
                raise RuntimeError('Native worker failed; inspect '+str(self.stderr_path)) from self.failure
            return None

    def release(self):
        self.process.stdin.write(b'release\n');self.process.stdin.flush()
        while True:
            result=self.take()
            if result is not None:
                if result['event']!='worker_released':
                    raise RuntimeError('Unexpected native worker release response')
                return
            time.sleep(0.005)

    def close(self):
        self.closing=True
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait()
        self.thread.join();self.process.stdout.close()
        if self.stderr is not None:
            self.stderr.close()
