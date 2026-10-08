"""Record the native build's inputs and executable, for provenance."""
import hashlib
import json
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
build=Path(sys.argv[1])
binary=build/"etazero"
cache=(build/"CMakeCache.txt").read_text()
with_torch="ETAZERO_WITH_TORCH:BOOL=ON" in cache
sources=[p for folder in ("src","include") for p in (root/"cpp"/folder).rglob("*") if p.is_file()]
sources.append(root/"cpp/CMakeLists.txt")
sources.append(root/"python"/"etazero"/"schema.py")
value={"with_torch":with_torch,"sources":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
       "binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest() if with_torch else None}
temporary=build/"build_manifest.json.tmp"
temporary.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
os.replace(temporary,build/"build_manifest.json")
