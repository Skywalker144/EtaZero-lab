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
    parser.add_argument('--version', help='版本目录名；省略时选择最高版本')
    options, remaining = parser.parse_known_args()
    lab = Path(__file__).resolve().parents[1]
    if options.version:
        if not re.fullmatch(r'EtaZero_V\d+(?:\.\d+)*', options.version):
            parser.error('--version 必须是 EtaZero_V<版本号> 目录名')
        root = lab / options.version
        if not root.is_dir():
            parser.error(f'版本不存在：{options.version}')
    else:
        root = latest_version(lab)
    native = (root / 'python/etazero/web_app.py').is_file()
    if not native and not (root / 'python/etazero/evaluation/analysis.py').is_file():
        parser.error(f'{root.name} 尚未接入 Web 协议。可显式使用 --version EtaZero_V3.4。')
    sys.path.insert(0, str(root / 'python'))
    os.environ['PYTHONPATH'] = os.pathsep.join([str(root / 'python'), str(lab), os.environ.get('PYTHONPATH', '')])
    if native:
        from web.native_server import main as serve
    else:
        from web.server import main as serve

    args = remaining + (['--binary', str(options.binary)] if options.binary else [])
    if '--help' in remaining or '-h' in remaining:
        print('web/webui.sh --rebuild：强制重新检查并构建本版本原生程序。', flush=True)
        return serve(args)
    if options.binary is None:
        build_dir = root / 'build'
        if options.rebuild and build_dir.is_dir():
            subprocess.run(['cmake', '--build', str(build_dir), '--target', 'clean'], check=True)
        print(f'EtaZero Web: 正在构建 {root.name}', flush=True)
        subprocess.run(['bash', str(root / 'scripts/build.sh')], check=True)
    elif options.rebuild:
        parser.error('--rebuild 不能与 --binary 同时使用')
    serve(args)


if __name__ == '__main__':
    main()
