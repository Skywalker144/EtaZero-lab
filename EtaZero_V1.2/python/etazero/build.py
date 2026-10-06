"""Check native build capability; CMake owns source dependency tracking."""
from pathlib import Path
from .storage import load_json, sha256


def verify_build(binary):
    binary = Path(binary)
    manifest = load_json(binary.parent/'build_manifest.json')
    if not binary.is_file() or not manifest['with_torch']:
        raise ValueError('Native executable requires a LibTorch build; run scripts/build.sh')
    # Identity for provenance, with no stale-source or checksum gate.
    return sha256(binary)
