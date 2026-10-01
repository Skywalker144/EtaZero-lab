"""Durable atomic publication; immutable payloads and small mutable pointers."""
import hashlib
import json
import os
from pathlib import Path
import uuid


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path, writer, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        writer(temporary)
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        if immutable:
            # link, unlike replace, atomically refuses to overwrite an existing artifact.
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value, immutable=False):
    atomic_write(path, lambda p: p.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n"), immutable)


def load_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path, arrays):
    import numpy as np
    def write(p):
        with p.open("wb") as file:
            np.savez_compressed(file, **arrays)
    atomic_write(path, write, immutable=True)


# Context-manager form used by the standalone arena and Elo writer.
from contextlib import contextmanager
import fcntl


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    try:
        yield temporary
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process owns this output directory') from error
        yield


write_json = save_json
