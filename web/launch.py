"""Choose the latest version; CMake handles incremental native builds."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


def latest_version(root):
    versions = [path for path in root.glob('EtaZero_V*')
                if path.is_dir() and re.fullmatch(r'EtaZero_V\d+(?:\.\d+)*', path.name)]
    return max(versions, key=lambda path: tuple(map(int, path.name.removeprefix('EtaZero_V').split('.'))))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--binary', type=Path)
    options, remaining = parser.parse_known_args()
    lab = Path(__file__).resolve().parents[1]
    root = latest_version(lab)
    sys.path.insert(0, str(root / 'python'))
    os.environ['PYTHONPATH'] = os.pathsep.join([str(root / 'python'), str(lab), os.environ.get('PYTHONPATH', '')])
    from web.server import main as serve

    args = remaining + (['--binary', str(options.binary)] if options.binary else [])
    if '--help' in remaining or '-h' in remaining:
        print('web/webui.sh --rebuild：强制重新检查并构建本版本原生程序。', flush=True)
        return serve(args)
    if options.binary is None:
        print(f'EtaZero Web: 正在构建 {root.name}', flush=True)
        subprocess.run(['bash', str(root / 'scripts/build.sh')], check=True)
    elif options.rebuild:
        parser.error('--rebuild 不能与 --binary 同时使用')
    serve(args)


if __name__ == '__main__':
    main()
