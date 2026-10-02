import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from etazero.config import ROOT
from etazero.eval_config import load_evaluation_config
from etazero.runtime import verify_build
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
            files = {'/': 'index.html', '/app.js': 'app.js', '/styles.css': 'styles.css'}
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
    if not models:
        raise ValueError('未找到导出的模型，请用 --model 指定 TorchScript 模型')
    for path in models.values():
        info = load_json(path.parent / 'manifest.json')
        if info['contract'] != CONTRACT_ID or type(info['canvas']) is not int or not 5 <= info['canvas'] <= 25:
            raise ValueError(f'模型输入契约或棋盘尺寸不匹配：{path}')
    return models


def main():
    parser = argparse.ArgumentParser(description='EtaZero 开发工作台')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--config-dir', default=str(ROOT / 'configs/minimal_test'))
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'data/minimal_test')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/etazero')
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error('缺少 etazero，请先运行 web/webui.sh 构建')
    try:
        verify_build(args.binary)
        models = discover_models(args.models_dir, args.model)
        config = load_evaluation_config(args.config_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    app = App(args.binary, models, config, config['evaluation']['board_size'],
              discover_models=lambda: discover_models(args.models_dir, args.model))
    server = make_server(app, args.host, args.port)
    print(f'EtaZero Web: http://{args.host}:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == '__main__':
    main()
