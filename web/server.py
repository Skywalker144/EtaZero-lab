import argparse
import json
import mimetypes
from threading import Thread
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from etazero.config import ROOT
from etazero.engine_config import load_engine_config, load_match_opening_config
from etazero.build import verify_build
from etazero.schema import CONTRACT_ID
from etazero.storage import load_json
from .app import App, Conflict


STATIC = Path(__file__).resolve().parent / 'static'


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status: int, data: bytes, content_type: str):
            try:
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, status: int, value):
            self.send(status, json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == '/api/state':
                try:
                    since = int(parse_qs(url.query).get('since', ['-1'])[0])
                except ValueError:
                    return self.json(400, dict(error='无效版本号'))
                return self.json(200, app.snapshot(since))
            if url.path == '/api/catalog':
                return self.json(200, app.catalog())
            files = {'/': 'index.html', '/app.js': 'app.js', '/theme.js': 'theme.js', '/styles.css': 'styles.css'}
            if url.path in files:
                path = STATIC / files[url.path]
                return self.send(200, path.read_bytes(), (mimetypes.guess_type(path)[0] or 'text/plain') + '; charset=utf-8')
            self.json(404, dict(error='页面不存在'))

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                return self.json(403, dict(error='请求来源不匹配'))
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('无效请求长度')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('请求必须是对象')
                operations = {f'/api/{name}': name for name in
                              ('new', 'play', 'undo', 'retry', 'analyze', 'step', 'branch', 'configure', 'refresh')}
                if self.path not in operations:
                    return self.json(404, dict(error='未知接口'))
                self.json(202, app.submit(operations[self.path], payload))
            except Conflict as error:
                self.json(409, dict(error=str(error)))
            except (ValueError, TypeError) as error:
                self.json(400, dict(error=str(error)))

    return ThreadingHTTPServer((host, port), Handler)


def discover_models(models_dir: Path, model: Path | None = None) -> dict[str, Path]:
    # Publication stages use .tmp_* directories; offer only committed exports.
    paths = sorted(models_dir.glob('**/models/iteration_*/model.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    paths = [path for path in paths if not any(part.startswith('.') for part in path.relative_to(models_dir).parts)]
    models = {str(Path(models_dir.name) / path.relative_to(models_dir)): path.resolve() for path in paths}
    current = models_dir / 'models/current.json'
    if current.is_file():
        preferred = (models_dir / load_json(current)['model']['path']).resolve()
        key = next((key for key, path in models.items() if path == preferred), None)
        if key is not None:
            models = {key: models[key], **{k: p for k, p in models.items() if k != key}}
    if model:
        selected = model.resolve()
        if not selected.is_file():
            raise ValueError('指定的模型文件不存在')
        key = next((key for key, path in models.items() if path == selected), str(selected))
        models = {key: selected, **{key: path for key, path in models.items() if path != selected}}
    for path in models.values():
        info = load_json(path.parent / 'manifest.json')
        if info['contract'] != CONTRACT_ID or type(info['canvas']) is not int or not 5 <= info['canvas'] <= 25:
            raise ValueError(f'模型输入契约或棋盘尺寸不匹配：{path}')
        if info.get('algorithm', 'alphazero') not in ('alphazero', 'muzero'):
            raise ValueError(f'不支持的模型算法：{path}')
    return models


def discover_catalog(models_dir: Path, model: Path | None = None, config_dir: Path | None = None):
    models = discover_models(models_dir, model)
    # A run may have saved its configuration but not yet published its first model.
    roots = {path.parent.parent for path in models_dir.glob('**/config/effective.json')}
    roots.update(path.parent for path in models_dir.glob('**/models') if path.is_dir())
    roots = {path for path in roots if not any(part.startswith('.') for part in path.relative_to(models_dir).parts)}
    roots.update(path.parent.parent.parent for path in models.values())
    runs = {}
    for root in sorted(roots):
        root = root.resolve()
        try:
            label = str(root.relative_to((ROOT / 'data').resolve()))
            mapped = ROOT / 'configs' / label
        except ValueError:
            label, mapped = root.name, None
        paths = [path for path in models.values() if path.parent.parent.parent == root]
        saved = root / 'config/effective.json'
        info = load_json(paths[0].parent / 'manifest.json') if paths else {}
        effective = load_json(saved) if saved.is_file() else {}
        algorithm = info.get('algorithm', effective.get('agent', {}).get('algorithm', 'alphazero'))
        source = config_dir or (mapped if mapped and (mapped / 'analysis.cfg').is_file() else
                                ROOT / 'configs' / ('muzero' if algorithm == 'muzero' else 'baseline'))
        config = load_engine_config(source)
        if not config_dir and source != mapped:
            config['analysis']['board_size'] = info.get('canvas', effective.get('network', {}).get('canvas', 15))
        runs[str(root)] = dict(id=str(root), label=label, path=str(root), algorithm=algorithm,
                               config_dir=str(source.resolve()), analysis_config=config['analysis'],
                               opening=load_match_opening_config(source))
        if model and model.resolve().parent.parent.parent == root:
            runs[str(root)]['selected_model'] = next(key for key, path in models.items() if path == model.resolve())
    return models, runs


def main(argv=None):
    parser = argparse.ArgumentParser(description='EtaZero 开发工作台')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--config-dir', type=Path, help='统一覆盖各配置目录的评估配置；默认使用对应配置')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/etazero')
    parser.add_argument('--no-open', action='store_true', help='不自动打开浏览器')
    args = parser.parse_args(argv)
    if not args.binary.is_file():
        parser.error('缺少 etazero，请先运行 web/webui.sh 构建')
    try:
        verify_build(args.binary)
        models, runs = discover_catalog(args.models_dir, args.model, args.config_dir)
        config = load_engine_config(args.config_dir or ROOT / 'configs/minimal_test')
        config['opening'] = load_match_opening_config(args.config_dir or ROOT / 'configs/minimal_test')
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    app = App(args.binary, models, config, config['analysis']['board_size'],
              runs=runs, discover_catalog=lambda: discover_catalog(args.models_dir, args.model, args.config_dir))
    server = make_server(app, args.host, args.port)
    browser_host = '127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host
    if ':' in browser_host:
        browser_host = f'[{browser_host}]'
    url = f'http://{browser_host}:{server.server_port}'
    print(f'EtaZero Web: {url}', flush=True)
    if not args.no_open:
        Thread(target=open_browser, args=(url,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


def open_browser(url):
    try:
        if webbrowser.open(url, new=2):
            return
    except webbrowser.Error:
        pass
    print(f'未能自动打开浏览器，请手动访问 {url}', flush=True)


if __name__ == '__main__':
    main()
