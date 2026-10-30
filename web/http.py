"""Shared local HTTP transport, independent of any engine or model format."""
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


def make_server(app, host: str, port: int, *, files, conflict, max_request_bytes=16384,
                actions=('new', 'play', 'undo', 'retry', 'analyze', 'step', 'branch', 'configure', 'refresh', 'stop')) -> ThreadingHTTPServer:
    operations = {f'/api/{name}': name for name in actions}
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
            if url.path in files:
                path = files[url.path]
                return self.send(200, path.read_bytes(), (mimetypes.guess_type(path)[0] or 'text/plain') + '; charset=utf-8')
            self.json(404, dict(error='页面不存在'))

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                return self.json(403, dict(error='请求来源不匹配'))
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= max_request_bytes:
                    raise ValueError(f'请求长度须为 1 至 {max_request_bytes} 字节')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('请求必须是对象')
                if self.path not in operations:
                    return self.json(404, dict(error='未知接口'))
                self.json(202, app.submit(operations[self.path], payload))
            except conflict as error:
                self.json(409, dict(error=str(error)))
            except (ValueError, TypeError) as error:
                self.json(400, dict(error=str(error)))

    return ThreadingHTTPServer((host, port), Handler)
