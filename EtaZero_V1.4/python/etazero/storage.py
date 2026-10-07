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


def atomic_write(path, writer, immutable=False, *, durable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        writer(temporary)
        if durable:
            with temporary.open("rb") as file:
                os.fsync(file.fileno())
        if immutable:
            # link, unlike replace, atomically refuses to overwrite an existing artifact.
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        if durable:
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
    atomic_write(path, lambda p: write_npz(p, arrays), immutable=True)


def write_npz(path, arrays, compressed=True):
    """Stream compact arrays with fast DEFLATE; callers choose publication durability."""
    import numpy as np
    import zipfile
    mode = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, 'w', compression=mode, compresslevel=1 if compressed else None) as archive:
        for key, array in arrays.items():
            # Fixed timestamps make derived cache bytes independent of wall clock.
            info = zipfile.ZipInfo(key+'.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = mode
            info._compresslevel = 1 if compressed else None
            with archive.open(info, 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)


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
