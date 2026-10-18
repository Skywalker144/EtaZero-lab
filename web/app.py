from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from secrets import randbits
from threading import Condition, Event
import time
from typing import Any
from uuid import uuid4

from etazero.schema import RULES
from etazero.config import validate_hex_symmetry
from etazero.storage import load_json
from .engine import Engine


class Conflict(ValueError):
    pass


def integer(payload: dict, name: str, minimum: int, maximum: int) -> int:
    value = payload.get(name)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} 必须是 {minimum}–{maximum} 之间的整数')
    return value


def human_turns(game: dict, human: int):
    prefix = len((game.get('opening') or {}).get('moves', []))
    return [i for i in range(prefix, len(game['moves'])) if (1 if i % 2 == 0 else -1) == human]


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
                analysis_config=deepcopy(config['analysis']), opening=deepcopy(config.get('opening')),
                hex_opening=deepcopy(config.get('hex_opening'))) for key, path in models.items()}
        self.discover_catalog = discover_catalog
        self.engine: Engine | None = None
        self.condition = Condition()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='etazero-game')
        self.closed = False
        self.cancelled = Event()
        self.operation_version = 0
        self.state = dict(instance=uuid4().hex, version=0, catalog_revision=0, game_id=None, busy=False, phase='idle', game=None, analysis=None,
                          error=None, model=None, human=1, rule='freestyle',
                          visits=config['analysis']['visits'], mode='play', started_at=None, analysis_config=None,
                          opening_kind='empty', opening_config=None, analysis_progress=None, stop_requested=False,
                          analysis_enabled=False)

    def catalog(self) -> dict:
        c = self.config['analysis']
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
                               next((run['id'] for run in available if run['label'] == 'checks/minimal_test'),
                                    available[0]['id'] if available else runs[0]['id'] if runs else None))
        return dict(models=models, runs=runs, default_run=default_run, version=self.binary.parent.parent.name,
                    rules=RULES, default_rule=c['rule'],
                    default_size=self.default_size, default_visits=c['visits'],
                    search_threads=c['search_threads'], virtual_loss=c['virtual_loss'], device=c['device'],
                    analysis_config=deepcopy(c))

    def snapshot(self, since: int = -1) -> dict:
        with self.condition:
            if since >= 0:
                self.condition.wait_for(lambda: self.closed or self.state['version'] != since, timeout=20)
            return deepcopy(self.state)

    def publish(self, **values):
        with self.condition:
            self.state.update(values)
            self.state['version'] += 1
            # Search reports do not invalidate actions against the same position.
            if set(values) != {'analysis_progress'}:
                self.operation_version = self.state['version']
            self.condition.notify_all()

    def submit(self, operation: str, payload: dict) -> dict:
        with self.condition:
            if type(payload.get('version')) is not int:
                raise ValueError('缺少有效的棋局版本号')
            if operation == 'stop':
                if (self.closed or (self.state['phase'] not in ('thinking', 'analyzing', 'stopping') and
                                    not self.state['analysis_enabled']) or
                        payload.get('game_id') != self.state['game_id']):
                    raise Conflict('当前没有可停止的搜索')
                if not self.state['busy'] and self.state['phase'] == 'idle':
                    self.publish(analysis_enabled=False)
                    return deepcopy(self.state)
                self.cancelled.set()
                self.publish(analysis_enabled=False, busy=True, stop_requested=True, phase='stopping')
                if self.engine:
                    self.engine.stop()
                return deepcopy(self.state)
            if (self.closed or self.state['busy'] or
                    not self.operation_version <= payload['version'] <= self.state['version']):
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
                if payload.get('opening', 'empty') not in ('empty', 'balanced'):
                    raise ValueError('未知开局类型')
            elif operation == 'play':
                if not game or game['finished'] or (self.state['mode'] == 'play' and game['player'] != self.state['human']):
                    raise Conflict('当前不能落子')
                action = integer(payload, 'action', 0, game['board_size'] ** 2 - 1)
                if game['board'][action]:
                    raise ValueError('此处已有棋子')
            elif operation == 'undo':
                if not game or not game['moves'] or (self.state['mode'] == 'play' and not human_turns(game, self.state['human'])):
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
            if operation == 'analyze' and self.state['analysis_enabled']:
                raise Conflict('持续分析已经开启')
            self.cancelled.set()
            if self.state['phase'] == 'analyzing' and self.engine:
                self.engine.stop()
            self.cancelled = Event()
            self.publish(busy=True, phase='loading' if operation in ('new', 'refresh') else 'thinking',
                         error=None, started_at=time.time(), analysis_progress=None, stop_requested=False,
                         analysis_enabled=operation == 'analyze' or self.state['analysis_enabled'])
            self.executor.submit(self.perform, operation, dict(payload), self.cancelled)
            return deepcopy(self.state)

    def perform(self, operation: str, payload: dict, cancelled: Event):
        try:
            if operation == 'new':
                selected = payload['model']
                run = self.runs[str(self.models[selected].resolve().parent.parent.parent)]
                config = {'analysis': deepcopy(run['analysis_config'])}
                opening_kind = payload.get('opening', 'empty')
                if run.get('opening') is not None:
                    config['opening'] = deepcopy(run['opening'])
                if run.get('hex_opening') is not None:
                    config['hex_opening'] = deepcopy(run['hex_opening'])
                if payload['rule']=='hex': validate_hex_symmetry(config['analysis'])
                if opening_kind=='balanced' and payload['rule']=='hex' and 'hex_opening' not in config:
                    raise ValueError('所选配置缺少 match.cfg Hex 平衡开局参数')
                if opening_kind == 'balanced' and 'opening' not in config:
                    raise ValueError('所选配置缺少 match.cfg 平衡开局参数')
                opening_profile=config.get('hex_opening' if payload['rule']=='hex' else 'opening')
                replacement = (self.engine is None or self.engine.process.poll() is not None or
                               selected != self.state['model'] or config['analysis'] != self.state['analysis_config'] or
                               opening_profile != self.state['opening_config'])
                engine = Engine(self.binary, self.models[selected], config) if replacement else self.engine
                try:
                    if payload['size'] > engine.canvas:
                        raise ValueError(f'该模型最大支持 {engine.canvas}×{engine.canvas} 棋盘')
                    command = f"new {payload['size']} {payload['rule']}"
                    if opening_kind == 'balanced':
                        self.publish(phase='opening', started_at=time.time())
                        # Let native cancellation finish before the transport's hard timeout.
                        timeout_ms = max(1, min(120000, int((engine.timeout - 5) * 1000)))
                        command += f' balanced {randbits(64)} {timeout_ms}'
                    reply = engine.command(command)
                except BaseException:
                    if replacement:
                        engine.close()
                    raise
                previous, self.engine = self.engine, engine
                if replacement and previous:
                    previous.close()
                self.publish(game=reply['state'], game_id=uuid4().hex, analysis=None, model=selected,
                             human=payload['human'], rule=payload['rule'], visits=payload['visits'],
                             mode=payload.get('mode', 'play'), analysis_config=deepcopy(config['analysis']),
                             opening_kind=opening_kind, opening_config=deepcopy(opening_profile))
            elif operation == 'play':
                reply = self.engine.command(f"play {payload['action']}")
                # Keep the last complete analysis visible until the next search succeeds.
                self.publish(game=reply['state'])
            elif operation == 'undo':
                moves = self.state['game']['moves']
                last_human = len(moves) - 1 if self.state['mode'] == 'manual' else max(human_turns(self.state['game'], self.state['human']))
                reply = self.engine.command(f'undo {len(moves) - last_human}')
                self.publish(game=reply['state'], analysis=None)
            elif operation == 'branch':
                reply = self.engine.command(f"undo {len(self.state['game']['moves']) - payload['turn']}")
                self.publish(game=reply['state'], analysis=None, mode='manual')
            elif operation == 'configure':
                self.publish(visits=payload['visits'], human=payload['human'], mode=payload.get('mode', 'play'))
            elif operation == 'step':
                self.search('genmove', cancelled)
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
                self.search('genmove', cancelled)
            if cancelled is not self.cancelled:
                return
            game = self.state['game']
            if self.state['analysis_enabled'] and not cancelled.is_set() and game and not game['finished']:
                self.search('analyze', cancelled)
            if cancelled is not self.cancelled:
                return
            self.publish(busy=False, phase='idle', started_at=None, stop_requested=False)
        except Exception as error:
            if cancelled is self.cancelled:
                self.publish(busy=False, phase='error', error=str(error), started_at=None,
                             analysis_progress=None, stop_requested=False, analysis_enabled=False)

    def search(self, command: str, cancelled: Event):
        continuous = command == 'analyze'
        planes = None
        if continuous:
            planes = self.engine.command('raw_nn')['network_planes']
            with self.condition:
                if cancelled.is_set():
                    return
                self.publish(busy=False, phase='analyzing', started_at=time.time())

        def report(analysis):
            with self.condition:
                if cancelled is self.cancelled and not cancelled.is_set():
                    self.publish(analysis_progress={**analysis, **({'network_planes': planes} if planes else {})})

        request = dict(command=command, visits=self.state['visits'], continuous=continuous)
        reply = self.engine.command(request, on_report=report, should_stop=cancelled.is_set)
        with self.condition:
            if cancelled is not self.cancelled:
                return
            if reply.get('cancelled') and 'analysis' not in reply:
                self.publish(analysis_progress=None)
                return
            analysis = reply['analysis']
            if planes:
                analysis = {**analysis, 'network_planes': planes}
            self.publish(game=reply['state'], analysis=analysis if analysis['completed_visits'] else self.state['analysis'],
                         analysis_progress=None)

    def close(self):
        with self.condition:
            self.closed = True
            self.cancelled.set()
            self.condition.notify_all()
        if self.engine:
            self.engine.stop()
        self.executor.shutdown(wait=True)
        if self.engine:
            self.engine.close()
