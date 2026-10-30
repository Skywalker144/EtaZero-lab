"""Asynchronous workbench control; native sessions own rules and analysis."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Condition, Event
from uuid import uuid4

from . import ROOT
from .web_session import Session


class Conflict(ValueError):
    """The browser action refers to a superseded or currently busy game."""


ACTION_FIELDS = {
    'new': {'model', 'size', 'rules', 'komi', 'fen', 'sgf', 'sgf_turn', 'opening', 'parameters', 'human', 'mode'},
    'configure': {'parameters', 'human', 'mode'},
    'play': {'move'}, 'branch': {'turn'},
    'diagnose': {'source', 'symmetry', 'optimism'},
    'refresh': set(), 'undo': set(), 'step': set(), 'retry': set(), 'analyze': set(), 'stop': set(),
}


class App:
    def __init__(self, discover_catalog, *, binary=None, human_model=None):
        self.discover_catalog = discover_catalog
        self._catalog = discover_catalog()
        self.binary, self.human_model = binary, human_model
        self.session = None
        self.preparing = None
        self.condition = Condition()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='etazero-gtp')
        self.cancelled = Event()
        self.closed = False
        self.operation_version = 0
        self.state = dict(instance=uuid4().hex, version=0, catalog_revision=0, game_id=None,
                          busy=False, phase='idle', game=None, history=[], resignation=None,
                          analysis=None, raw_nn=None, opening=None, parameters=None, models=[], config=None, selected_model=None,
                          human='B', mode='play', error=None, analysis_enabled=False, stop_requested=False,
                          connected=False)

    def catalog(self):
        from katago.game.xiangqi import MOVES
        with self.condition:
            return dict(deepcopy(self._catalog), version=ROOT.name, xiangqi_moves=MOVES)

    def snapshot(self, since=-1):
        with self.condition:
            if since >= 0:
                self.condition.wait_for(lambda: self.closed or self.state['version'] != since, timeout=20)
            return deepcopy(self.state)

    def _publish(self, *, token=None, progress=False, **values):
        with self.condition:
            if self.closed or (token is not None and token is not self.cancelled):
                return
            self.state.update(values)
            self.state['version'] += 1
            if not progress:
                self.operation_version = self.state['version']
            self.condition.notify_all()

    def submit(self, operation, payload):
        if not isinstance(payload, dict) or type(payload.get('version')) is not int:
            raise ValueError('A browser action requires an integer state version')
        if operation not in ACTION_FIELDS:
            raise ValueError('Unknown workbench action')
        unknown = payload.keys() - ACTION_FIELDS[operation] - {'version', 'game_id'}
        if unknown:
            raise ValueError('Unsupported workbench fields: ' + ', '.join(sorted(unknown)))
        with self.condition:
            if self.closed:
                raise Conflict('The workbench has closed')
            if operation == 'stop':
                if (payload.get('game_id') != self.state['game_id'] or
                        self.state['phase'] not in ('thinking', 'analyzing', 'opening', 'stopping')):
                    raise Conflict('There is no active search or opening generation for this game')
                self.cancelled.set()
                self._publish(analysis_enabled=False, busy=True, phase='stopping', stop_requested=True)
                active = self.preparing if self.preparing is not None else self.session
                if active is not None:
                    active.stop()
                return deepcopy(self.state)
            if self.state['busy'] or not self.operation_version <= payload['version'] <= self.state['version']:
                raise Conflict('The game has changed; wait for its current state')
            if operation not in ('new', 'refresh') and payload.get('game_id') != self.state['game_id']:
                raise Conflict('This action belongs to another game')
            if operation not in ('new', 'refresh') and self.session is not None and not self.state['connected']:
                raise Conflict('The native process disconnected; create a new game')
            payload = deepcopy(payload)
            game = self.state['game']
            finished = game and (game['isGameFinished'] or self.state['resignation'] is not None)
            if operation in ('new', 'configure'):
                payload.setdefault('human', self.state['human'])
                payload.setdefault('mode', self.state['mode'])
                if payload['human'] not in ('B', 'W') or payload['mode'] not in ('manual', 'play'):
                    raise ValueError('Choose B/W and manual/play mode')
                if 'parameters' in payload and not isinstance(payload['parameters'], dict):
                    raise ValueError('parameters must be a native parameter object')
            if operation == 'new':
                if 'opening' in payload:
                    if not isinstance(payload['opening'], dict):
                        raise ValueError('opening must be a native opening options object')
                    if any(payload.get(key) is not None for key in ('sgf', 'sgf_turn', 'fen')):
                        raise ValueError('A balanced opening cannot be combined with SGF or FEN')
                selected = next(((run, model) for run in self._catalog['runs'] for model in run['models']
                                 if model['id'] == payload.get('model')), None)
                if selected is None:
                    raise ValueError('Select a published model')
                run, model = selected
                if run['config'] is None:
                    raise ValueError('The selected run needs a native gtp.cfg')
                payload['_model'], payload['_config'] = deepcopy(model), run['config']
            elif operation == 'play':
                if not game or finished or (self.state['mode'] == 'play' and game['currentPlayer'] != self.state['human']):
                    raise Conflict('A human move is not available in this position')
                if payload.get('move') not in game['legalMoves']:
                    raise ValueError('Select a native legal move')
            elif operation == 'undo':
                if not game:
                    raise Conflict('Create a game before undoing a move')
                moves = game['moves']
                if self.state['mode'] == 'manual':
                    count = min(1, len(moves))
                else:
                    turns = [i for i, (player, _) in enumerate(moves) if player == self.state['human']]
                    count = len(moves) - turns[-1] if turns else 0
                if count == 0 and self.state['resignation'] is None:
                    raise ValueError('There is no move to undo')
                payload['_count'] = count
            elif operation == 'branch':
                turn = payload.get('turn')
                if not game or type(turn) is not int or not 0 <= turn < len(self.state['history']):
                    raise ValueError('Select a position in the current session history')
                payload['_count'] = len(self.state['history']) - 1 - turn
                if payload['_count'] == 0 and self.state['resignation'] is None:
                    raise ValueError('The selected position is already current')
            elif operation in ('analyze', 'step', 'retry', 'diagnose'):
                if not game or finished:
                    raise Conflict('There is no unfinished position to search')
                if operation == 'analyze' and self.state['analysis_enabled']:
                    raise Conflict('Continuous analysis is already enabled')
                if operation == 'retry' and (self.state['mode'] != 'play' or game['currentPlayer'] == self.state['human']):
                    raise Conflict('It is not the AI turn')
            elif operation == 'configure':
                if not game:
                    raise Conflict('Create a game before changing its parameters')
            elif operation != 'refresh':
                raise ValueError('Unknown workbench action')

            self.cancelled.set()
            if self.state['phase'] == 'analyzing' and self.session is not None:
                self.session.stop()
            self.cancelled = Event()
            phase = ('loading' if operation in ('new', 'refresh') else
                     'thinking' if operation in ('step', 'retry') else
                     'diagnosing' if operation == 'diagnose' else
                     'analyzing' if operation == 'analyze' else 'working')
            self._publish(busy=True, phase=phase,
                          error=None, stop_requested=False,
                          analysis_enabled=operation == 'analyze' or self.state['analysis_enabled'])
            self.executor.submit(self._perform, operation, payload, self.cancelled)
            return deepcopy(self.state)

    def _search(self, token, *, continuous=False, play=False):
        def report(_):
            if not token.is_set():
                self._publish(token=token, progress=True, analysis=self.session.analysis_snapshot())
        self.session.search(continuous=continuous, play=play, cancelled=token, on_report=report)
        self._publish(token=token, **self.session.snapshot())

    def _perform(self, operation, payload, token):
        candidate = None
        try:
            if token.is_set():
                return
            if operation == 'new':
                candidate = Session(payload['_config'], payload['_model']['path'], binary=self.binary,
                                    human_model=self.human_model,
                                    **{k: payload[k] for k in ('size', 'rules', 'komi', 'fen', 'sgf', 'sgf_turn', 'parameters') if k in payload})
                if 'opening' in payload:
                    with self.condition:
                        if self.closed or token.is_set():
                            return
                        self.preparing = candidate
                        self._publish(token=token, phase='opening')
                    if not candidate.generate_opening(payload['opening'], cancelled=token):
                        return
                with self.condition:
                    if self.closed or token.is_set():
                        return
                    previous, self.session = self.session, candidate
                    self.preparing = None
                    candidate = None
                    self._publish(token=token, **self.session.snapshot(), game_id=uuid4().hex,
                                  selected_model=payload['_model'], human=payload['human'], mode=payload['mode'])
                if previous is not None:
                    previous.close()
            elif operation == 'play':
                self.session.play(payload['move'])
                self._publish(token=token, **self.session.snapshot())
            elif operation in ('undo', 'branch'):
                self.session.undo(payload['_count'])
                self._publish(token=token, **self.session.snapshot(),
                              **(dict(mode='manual') if operation == 'branch' else {}))
            elif operation == 'configure':
                if 'parameters' in payload:
                    self.session.configure(payload['parameters'])
                self._publish(token=token, **self.session.snapshot(), human=payload['human'], mode=payload['mode'])
            elif operation == 'step':
                self._search(token, play=True)
            elif operation == 'diagnose':
                self.session.diagnose(**{k: payload[k] for k in ('source', 'symmetry', 'optimism') if k in payload})
                self._publish(token=token, **self.session.snapshot())
            elif operation == 'refresh':
                catalog = self.discover_catalog()
                with self.condition:
                    self._catalog = catalog
                    self._publish(token=token, catalog_revision=self.state['catalog_revision'] + 1)

            if token.is_set():
                return
            if (operation in ('new', 'play', 'retry') and self.state['mode'] == 'play' and
                    not self.session.finished and self.session.state['currentPlayer'] != self.state['human']):
                self._publish(token=token, phase='thinking')
                self._search(token, play=True)
            if token.is_set():
                return
            if self.session is not None and self.state['analysis_enabled'] and not self.session.finished:
                self._publish(token=token, busy=False, phase='analyzing')
                self._search(token, continuous=True)
        except Exception as error:
            known = self.session.snapshot() if self.session is not None else {}
            self._publish(token=token, **known, error=str(error), phase='error', analysis_enabled=False)
        finally:
            if candidate is not None:
                candidate.close()
            with self.condition:
                self.preparing = None
                if not self.closed and token is self.cancelled:
                    self._publish(token=token, busy=False,
                                  phase='error' if self.state['error'] else 'idle',
                                  stop_requested=False, analysis_enabled=False)

    def close(self):
        with self.condition:
            self.closed = True
            self.cancelled.set()
            if self.session is not None:
                self.session.stop()
            if self.preparing is not None:
                self.preparing.stop()
            self.condition.notify_all()
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.session is not None:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
