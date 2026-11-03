"""Subprocess harness for production match dispatch and cancellation checks."""
from pathlib import Path
import signal
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from etazero.evaluation import execute_match, validate_model_window


def worker_command(command, capacity=None):
    return [sys.executable, str(Path(__file__).resolve()), str(capacity), *command]


def interrupt(signum, frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    capacity = None if sys.argv[1] == 'None' else int(sys.argv[1])
    validate_model_window(capacity)
    signal.signal(signal.SIGTERM, interrupt)
    with tempfile.TemporaryDirectory(prefix='etazero-dispatch-test-') as temporary:
        log = Path(temporary) / 'match.log'
        status = 0
        try:
            execute_match(sys.argv[2:], Path.cwd(), log, (), None, max_loaded_models=capacity)
        except KeyboardInterrupt:
            # execute_match terminates and waits for the native writer first.
            pass
        except Exception as error:
            print(error, file=sys.stderr)
            status = 1
        finally:
            if log.exists():
                print(log.read_text(), end='')
        sys.exit(status)
