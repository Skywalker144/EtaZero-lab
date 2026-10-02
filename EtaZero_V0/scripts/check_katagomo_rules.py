"""Compile the fixed KataGomo Renju oracle independently; no Torch/GPU required."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'df152116e3787c75c6a3de099d261ca092b7dfc1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/sky/RL/SkyZero/KataGomo'))
    args = parser.parse_args()
    source = args.source.resolve()
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('KataGomo commit differs from the fixed reference')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('KataGomo working tree must be clean')
    with tempfile.TemporaryDirectory(prefix='etazero_rules_oracle_') as directory:
        build = Path(directory)
        subprocess.run([sys.executable, str(ROOT/'python/etazero/schema.py'), str(build/'etazero/schema.h')], check=True)
        finder = source/'cpp/forbiddenPoint'
        subprocess.run(['g++', '-std=c++17', '-O2', '-I', str(finder), '-I', str(ROOT/'cpp/include'), '-I', str(build),
                        str(ROOT/'cpp/tests/rules_reference.cpp'), str(finder/'ForbiddenPointFinder.cpp'),
                        str(ROOT/'cpp/src/game/rules.cpp'), '-o', str(build/'rules_oracle')], check=True)
        result = json.loads(subprocess.check_output([str(build/'rules_oracle')], text=True, timeout=120))
    result.update(commit=COMMIT, source_sha256={path: hashlib.sha256((source/path).read_bytes()).hexdigest()
                  for path in ('cpp/forbiddenPoint/ForbiddenPointFinder.cpp','cpp/forbiddenPoint/ForbiddenPointFinder.h')},
                  limits='Finite boundary/pattern corpus; not a proof for every recursive board state. No VCN, rectangular board or pass semantics.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
