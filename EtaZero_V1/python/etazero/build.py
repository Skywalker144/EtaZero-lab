"""Native build verification without importing the training stack."""
from pathlib import Path

from .config import ROOT
from .storage import load_json, sha256


def verify_build(binary):
    binary = Path(binary)
    manifest = load_json(binary.parent / 'build_manifest.json')
    if not manifest['with_torch'] or sha256(binary) != manifest['binary_sha256']:
        raise ValueError('Native executable differs from its build manifest; rebuild before running')
    for path, checksum in manifest['sources'].items():
        if sha256(ROOT / path) != checksum:
            raise ValueError(f'Native build is stale for {path}; run scripts/build.sh')
    return manifest['binary_sha256']
