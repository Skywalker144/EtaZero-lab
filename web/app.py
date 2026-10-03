from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Condition
import time
from typing import Any
from uuid import uuid4

from etazero.schema import RULES
from etazero.storage import load_json
from .engine import Engine


class Conflict(ValueError):
    pass


def integer(payload: dict, name: str, minimum: int, maximum: int) -> int:
    value = payload.get(name)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} 必须是 {minimum}–{maximum} 之间的整数')
    return value


class App:
    def __init__(self, binary: Path, models: dict[str, Path], config: dict[str, Any], default_size: int,
                 runs=None, discover_catalog=None):
        self.binary, self.models, self.config = binary, models, config
        self.metadata = {key: load_json(path.parent / 'manifest.json') for key, path in models.items()}
        self.default_size = default_size
        self.runs = runs if runs is not None else {
            str(path.resolve().parent.parent.parent): dict(
                id=str(path.resolve().parent.parent.parent), label=path.resolve().parent.parent.parent.name,
                path=str(path.resolve().parent.parent.parent), algorithm=self.metadata[key].get('algorithm', 'alphazero'),
                evaluation=deepcopy(config['evaluation'])) for key, path in models.items()}
        self.discover_catalog = discover_catalog
        self.engine: Engine | None = None
        self.condition = Condition()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='etazero-game')
        self.closed = False
        self.state = dict(instance=uuid4().hex, version=0, catalog_revision=0, game_id=None, busy=False, phase='idle', game=None, analysis=None,
                          error=None, model=None, human=1, rule='freestyle',
                          visits=config['evaluation']['visits'], mode='play', started_at=None, evaluation=None)

    def catalog(self) -> dict:
        c = self.config['evaluation']
        with self.condition:
            models = [dict(id=key, run=str(self.models[key].resolve().parent.parent.parent),
                           iteration=info['checkpoint']['iteration'],
                           label=f"{self.models[key].parent.parent.parent.name} · 第 {info['checkpoint']['iteration']} 代 · {info.get('algorithm', 'alphazero')} · {info['canvas']}×{info['canvas']}",
                           canvas=info['canvas'], manifest=deepcopy(info), path=str(self.models[key]))
                      for key, info in self.metadata.items()]
            runs = deepcopy(list(self.runs.values()))
            for run in runs:
                generations = sorted((m for m in models if m['run'] == run['id']),
                                     key=lambda m: (m['iteration'], m['manifest']['checkpoint'].get('step', 0), m['id']), reverse=True)
                current = Path(run['path']) / 'models/current.json'
                preferred = None
                if current.is_file():
                    preferred = str((Path(run['path']) / load_json(current)['model']['path']).resolve())
                run['models'] = [m['id'] for m in generations]
                run['current_model'] = next((m['id'] for m in generations if m['path'] == preferred), None)
                run['default_model'] = run.get('selected_model') or run['current_model'] or (generations[0]['id'] if generations else None)
            available = [run for run in runs if run['models']]
            default_run = next((run['id'] for run in available if run.get('selected_model')),
                               next((run['id'] for run in available if run['label'] == 'minimal_test'),
                                    available[0]['id'] if available else runs[0]['id'] if runs else None))
        return dict(models=models, runs=runs, default_run=default_run, version=self.binary.parent.parent.name,
                    rules=RULES, default_rule=c['rule'],
                    default_size=self.default_size, default_visits=c['visits'],
                    search_threads=c['search_threads'], virtual_loss=c['virtual_loss'], device=c['device'],
                    evaluation=deepcopy(c))

    def snapshot(self, since: int = -1) -> dict:
        with self.condition:
            if since >= 0:
                self.condition.wait_for(lambda: self.closed or self.state['version'] != since, timeout=20)
            return deepcopy(self.state)

    def publish(self, **values):
        with self.condition:
            self.state.update(values)
            self.state['version'] += 1
            self.condition.notify_all()

    def submit(self, operation: str, payload: dict) -> dict:
        with self.condition:
            if type(payload.get('version')) is not int:
                raise ValueError('缺少有效的棋局版本号')
            if self.closed or self.state['busy'] or payload['version'] != self.state['version']:
                raise Conflict('棋局状态已更新，请等待同步后再操作')
            game = self.state['game']
            if operation in ('new', 'configure'):
                integer(payload, 'visits', 2, 100000)
                if type(payload.get('human')) is not int or payload['human'] not in (-1, 1):
                    raise ValueError('执子必须为黑或白')
                if payload.get('mode', 'play') not in ('play', 'manual'):
                    raise ValueError('未知操作模式')
                if operation == 'configure' and not game:
                    raise Conflict('请先创建棋局')
            if operation == 'new':
                if payload.get('model') not in self.models:
                    raise ValueError('请选择可用模型')
                integer(payload, 'size', 5, self.metadata[payload['model']]['canvas'])
                if payload.get('rule') not in RULES:
                    raise ValueError('未知棋规')
            elif operation == 'play':
                if not game or game['finished'] or (self.state['mode'] == 'play' and game['player'] != self.state['human']):
                    raise Conflict('当前不能落子')
                action = integer(payload, 'action', 0, game['board_size'] ** 2 - 1)
                if game['board'][action]:
                    raise ValueError('此处已有棋子')
            elif operation == 'undo':
                if not game or not game['moves'] or (self.state['mode'] == 'play' and not any(
                        (1 if i % 2 == 0 else -1) == self.state['human'] for i in range(len(game['moves'])))):
                    raise ValueError('还没有可以撤回的人类落子')
            elif operation == 'branch':
                if not game or not game['moves']:
                    raise Conflict('还没有可回退的棋局')
                integer(payload, 'turn', 0, len(game['moves']) - 1)
            elif operation in ('analyze', 'step'):
                if not game or game['finished']:
                    raise Conflict('当前没有可分析的局面')
            elif operation == 'retry':
                if self.state['mode'] != 'play' or not game or game['finished'] or game['player'] == self.state['human']:
                    raise Conflict('当前不需要 AI 落子')
            elif operation not in ('new', 'configure', 'refresh'):
                raise ValueError('未知操作')
            self.publish(busy=True, phase='loading' if operation in ('new', 'refresh') else 'thinking',
                         error=None, started_at=time.time())
            self.executor.submit(self.perform, operation, dict(payload))
            return deepcopy(self.state)

    def perform(self, operation: str, payload: dict):
        try:
            if operation == 'new':
                selected = payload['model']
                run = self.runs[str(self.models[selected].resolve().parent.parent.parent)]
                config = {'evaluation': deepcopy(run['evaluation'])}
                replacement = (self.engine is None or self.engine.process.poll() is not None or
                               selected != self.state['model'] or config['evaluation'] != self.state['evaluation'])
                engine = Engine(self.binary, self.models[selected], config) if replacement else self.engine
                try:
                    if payload['size'] > engine.canvas:
                        raise ValueError(f'该模型最大支持 {engine.canvas}×{engine.canvas} 棋盘')
                    reply = engine.command(f"new {payload['size']} {payload['rule']}")
                except BaseException:
                    if replacement:
                        engine.close()
                    raise
                previous, self.engine = self.engine, engine
                if replacement and previous:
                    previous.close()
                self.publish(game=reply['state'], game_id=uuid4().hex, analysis=None, model=selected,
                             human=payload['human'], rule=payload['rule'], visits=payload['visits'],
                             mode=payload.get('mode', 'play'), evaluation=deepcopy(config['evaluation']) if replacement else self.state['evaluation'])
            elif operation == 'play':
                reply = self.engine.command(f"play {payload['action']}")
                self.publish(game=reply['state'], analysis=None)
            elif operation == 'undo':
                moves = self.state['game']['moves']
                last_human = len(moves) - 1 if self.state['mode'] == 'manual' else max(
                    i for i in range(len(moves)) if (1 if i % 2 == 0 else -1) == self.state['human'])
                reply = self.engine.command(f'undo {len(moves) - last_human}')
                self.publish(game=reply['state'], analysis=None)
            elif operation == 'branch':
                reply = self.engine.command(f"undo {len(self.state['game']['moves']) - payload['turn']}")
                self.publish(game=reply['state'], analysis=None, mode='manual')
            elif operation == 'configure':
                self.publish(visits=payload['visits'], human=payload['human'], mode=payload.get('mode', 'play'))
            elif operation in ('analyze', 'step'):
                reply = self.engine.command(f"{'analyze' if operation == 'analyze' else 'genmove'} {self.state['visits']}")
                self.publish(game=reply['state'], analysis=reply['analysis'])
            elif operation == 'refresh' and self.discover_catalog:
                models, runs = self.discover_catalog()
                metadata = {key: load_json(path.parent / 'manifest.json') for key, path in models.items()}
                with self.condition:
                    # Keep the active model selectable even if its file was moved during this session.
                    active = self.state['model']
                    if active and active not in models:
                        models[active], metadata[active] = self.models[active], self.metadata[active]
                        root = str(self.models[active].resolve().parent.parent.parent)
                        runs.setdefault(root, self.runs[root])
                    self.models, self.metadata, self.runs = models, metadata, runs
                self.publish(catalog_revision=self.state['catalog_revision'] + 1)
            game = self.state['game']
            if operation in ('new', 'play', 'retry') and self.state['mode'] == 'play' and game and not game['finished'] and game['player'] != self.state['human']:
                self.publish(phase='thinking', started_at=time.time())
                reply = self.engine.command(f"genmove {self.state['visits']}")
                self.publish(game=reply['state'], analysis=reply['analysis'])
            self.publish(busy=False, phase='idle', started_at=None)
        except Exception as error:
            self.publish(busy=False, phase='error', error=str(error), started_at=None)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.executor.shutdown(wait=True)
        if self.engine:
            self.engine.close()
