"""V4 workbench entry: native GTP controller and source-backed browser views."""
import argparse
from pathlib import Path
from threading import Thread
import webbrowser

from etazero import ROOT
from etazero.web_app import ACTION_FIELDS, App, Conflict
from etazero.web_catalog import discover_catalog
from .http import make_server as make_http_server


STATIC = Path(__file__).resolve().parent / 'static'


def make_server(app, host, port):
    files = {'/': STATIC / 'native.html', '/app.js': STATIC / 'native.js',
             **{f'/{name}': STATIC / name for name in ('styles.css', 'native.css', 'theme.js')}}
    return make_http_server(app, host, port, files=files, conflict=Conflict,
                            max_request_bytes=4*1024*1024, actions=ACTION_FIELDS)


def catalog_for(directory, config=None, model=None):
    catalog = discover_catalog(directory, config=config)
    if model is not None:
        model = Path(model).resolve()
        if not model.exists():
            raise ValueError(f'Native model does not exist: {model}')
        key = str(model)
        found = any(row['id'] == key for run in catalog['runs'] for row in run['models'])
        if not found:
            if config is None:
                raise ValueError('A model outside a published run requires an explicit native --config')
            # Native loading checks the actual game, dimensions and model
            # format. An external model has no invented training metadata.
            catalog['runs'].insert(0, dict(id='direct-model', label='直接指定的原生模型', config=str(Path(config).resolve()),
                                         current_model=key, models=[dict(id=key, path=key, run='direct-model')]))
        catalog['default_model'] = key
    return catalog


def main(argv=None):
    parser = argparse.ArgumentParser(description='EtaZero V4 · KataGo 原生工作台')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--config', type=Path, help='统一使用的原生 GTP 配置文件')
    sources.add_argument('--config-dir', type=Path, help='统一使用该目录的 gtp.cfg')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--model', type=Path, help='选中的原生 .bin/.bin.gz 或 MuZero bundle')
    parser.add_argument('--human-model', type=Path, help='交给原生 GTP 的次级 HumanSL 模型')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/katago')
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args(argv)
    config = args.config_dir / 'gtp.cfg' if args.config_dir else args.config
    if not args.binary.is_file():
        parser.error('缺少本版本 katago，请先使用 web/webui.sh 构建')
    try:
        app = App(lambda: catalog_for(args.models_dir, config, args.model),
                  binary=args.binary, human_model=args.human_model)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        server = make_server(app, args.host, args.port)
        host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
        url = f'http://{host}:{server.server_port}'
        print(f'EtaZero Web ({ROOT.name}): {url}', flush=True)
        if not args.no_open:
            def open_browser():
                try:
                    if webbrowser.open(url, new=2):
                        return
                except webbrowser.Error:
                    pass
                print(f'请手动访问 {url}', flush=True)
            Thread(target=open_browser, daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            app.close()
            server.server_close()
    finally:
        app.close()


if __name__ == '__main__':
    main()
