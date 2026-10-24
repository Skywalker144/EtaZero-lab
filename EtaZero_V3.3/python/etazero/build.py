"""Locate the native executable; CMake owns build dependencies and capability."""
import os
from pathlib import Path


def verify_build(binary):
    if not Path(binary).is_file() or not os.access(binary, os.X_OK):
        raise ValueError('Native executable is missing or not executable; run scripts/build.sh')
