"""Workbench actions on one native GTP game, with source board snapshots.

Only ``stop`` and ``snapshot`` may run concurrently with an action. The Web
controller serializes the other methods; no Python board or search is kept.
"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock

from .gtp import GtpProcess
from .web_raw_nn import parse_raw_nn


def position(state):
    # Retain actual boards for captures and Xiangqi. Copying the ever-growing
    # move list into every history entry would take quadratic space.
    return {key: deepcopy(value) for key, value in state.items() if key not in ('moves', 'legalMoves')}


class Session:
    def __init__(self, config, model, *, binary=None, human_model=None, size=None,
                 rules=None, komi=None, fen=None, sgf=None, sgf_turn=None, parameters=None, timeout=120):
        if sgf is not None:
            if not isinstance(sgf, str) or not sgf.strip():
                raise ValueError('sgf must contain a nonempty SGF record')
            if any(value is not None for value in (size, rules, komi, fen)):
                raise ValueError('SGF supplies its own position; clear size, rules, komi and FEN overrides')
        if sgf_turn is not None and (sgf is None or type(sgf_turn) is not int or sgf_turn < 0):
            raise ValueError('sgf_turn requires an SGF and a nonnegative integer move count')
        self.lock = RLock()
        self.cancelled = None
        self.analysis = None
        self.raw_nn = None
        self.resignation = None
        self.opening = None
        self.config = str(Path(config).resolve())
        self.engine = GtpProcess(config, model, binary=binary, human_model=human_model, timeout=timeout)
        try:
            # A new Session is prepared before replacing the active one. Failed
            # multi-command setup never changes the user's existing game.
            if size is not None:
                if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(n) is not int for n in size):
                    raise ValueError('size must contain two integer board dimensions')
                self.engine.command(f'boardsize {size[0]} {size[1]}')
            if rules is not None:
                if not isinstance(rules, (str, dict)):
                    raise ValueError('rules must be a native rule name or JSON object')
                self.engine.command('kata-set-rules ' + (rules if isinstance(rules, str) else json.dumps(rules, allow_nan=False)))
            if komi is not None:
                if type(komi) not in (int, float):
                    raise ValueError('komi must be a number')
                self.engine.command('komi ' + json.dumps(komi, allow_nan=False))
            if fen is not None:
                if not isinstance(fen, str) or not fen.strip():
                    raise ValueError('fen must be a nonempty native FEN string')
                self.engine.command('set_position fen ' + fen)
            if sgf is not None:
                # Only the native reader parses SGF, chooses its main variation
                # and restores rule history. No client filename enters GTP.
                with TemporaryDirectory(prefix='etazero-web-sgf-', dir='/tmp') as directory:
                    path = Path(directory) / 'position.sgf'
                    path.write_text(sgf, encoding='utf-8')
                    suffix = '' if sgf_turn is None else f' {sgf_turn + 1}'
                    self.engine.command(f'loadsgf {path}{suffix}')
            self.state = self.engine.state()
            self.history = (json.loads(self.engine.command('kata-get-history').body) if sgf is not None
                            else [position(self.state)])
            if parameters is not None:
                self.configure(parameters)
            else:
                self.parameters = json.loads(self.engine.command('kata-get-params').body)
        except BaseException:
            self.engine.close()
            raise

    @property
    def finished(self):
        return self.state['isGameFinished'] or self.resignation is not None

    def snapshot(self):
        with self.lock:
            return deepcopy(dict(game=self.state, history=self.history, resignation=self.resignation,
                                 analysis=self.analysis, parameters=self.parameters,
                                 opening=self.opening, raw_nn=self.raw_nn,
                                 models=self.engine.models, config=self.config,
                                 connected=self.engine.process.poll() is None))

    def analysis_snapshot(self):
        with self.lock:
            return deepcopy(self.analysis)

    def configure(self, parameters):
        if not isinstance(parameters, dict):
            raise ValueError('parameters must be a native parameter object')
        encoded = json.dumps(parameters, allow_nan=False)
        with self.lock:
            self.engine.command('kata-set-params ' + encoded)
            self.parameters = json.loads(self.engine.command('kata-get-params').body)
            self.analysis = None
            self.raw_nn = None

    def diagnose(self, *, source='main', symmetry=0, optimism=None):
        if source not in ('main', 'human') or type(symmetry) is not int or not 0 <= symmetry <= 7:
            raise ValueError('Choose main/human and an integer symmetry in [0,7]')
        if source == 'human' and optimism is not None:
            raise ValueError('kata-raw-human-nn does not accept policy optimism')
        command = f'kata-raw-{"human-" if source == "human" else ""}nn {symmetry}'
        if source == 'main':
            optimism = 0 if optimism is None else optimism
            if type(optimism) not in (int, float) or not 0 <= optimism <= 1:
                raise ValueError('Raw-NN policy optimism must be in [0,1]')
            command += ' ' + json.dumps(optimism, allow_nan=False)
        with self.lock:
            if self.finished:
                raise ValueError('The game has finished')
            reply = self.engine.command(command)
            report = parse_raw_nn(reply.body, self.state)
            self.raw_nn = dict(position=position(self.state), ply=len(self.history) - 1,
                               source=source, command=command, text=reply.body, report=report)

    def play(self, move):
        if not isinstance(move, str) or not move or any(c.isspace() for c in move):
            raise ValueError('move must be a single native coordinate')
        with self.lock:
            if self.finished:
                raise ValueError('The game has finished')
            self.engine.command(f'play {self.state["currentPlayer"]} {move}')
            self.state = self.engine.state()
            self.history.append(position(self.state))

    def generate_opening(self, options, *, cancelled=None):
        """Prepare a new session through the native suggestion and play commands.

        The App publishes this candidate only after the entire prefix succeeds.
        Native rejection sampling has no retry/time cap; stop() cancels it.
        """
        if not isinstance(options, dict):
            raise ValueError('opening must be a native opening options object')
        encoded = json.dumps(options, allow_nan=False)
        with self.lock:
            cancelled = cancelled if cancelled is not None else Event()
            self.cancelled = cancelled
        try:
            reply = self.engine.command('kata-suggest-opening ' + encoded,
                                        should_stop=cancelled.is_set, unlimited_wait=True)
            suggestion = json.loads(reply.body)
            with self.lock:
                if cancelled.is_set() or suggestion['status'] == 'cancelled':
                    return False
                if suggestion['status'] != 'complete' or not suggestion['moves']:
                    raise RuntimeError('Native engine did not generate a balanced opening')
                for player, move in suggestion['moves']:
                    self.engine.command(f'play {player} {move}')
                self.state = self.engine.state()
                self.history = json.loads(self.engine.command('kata-get-history').body)
                self.analysis = None
                self.raw_nn = None
                self.opening = suggestion
                return True
        finally:
            with self.lock:
                self.cancelled = None

    def undo(self, count=1):
        with self.lock:
            if type(count) is not int or not 0 <= count < len(self.history):
                raise ValueError('undo count exceeds the current session history')
            if count == 0 and self.resignation is None:
                raise ValueError('There is no move or resignation to undo')
            for _ in range(count):
                self.engine.command('undo')
                self.state = self.engine.state()
                self.history.pop()
                # Tolerant SGF can contain consecutive same-player moves.
                # Source undo may reset rule history while replaying them;
                # the current snapshot must reflect the resulting native state.
                self.history[-1] = position(self.state)
            self.resignation = None
            self.analysis = None
            self.raw_nn = None

    def search(self, *, play=False, continuous=False, on_report=None, cancelled=None):
        if play and continuous:
            raise ValueError('Continuous analysis does not play a move')
        with self.lock:
            if self.finished:
                raise ValueError('The game has finished')
            cancelled = cancelled if cancelled is not None else Event()
            self.cancelled = cancelled
            self.analysis = dict(position=position(self.state), ply=len(self.history) - 1,
                                 report=None, status='running', move=None)
            player = self.state['currentPlayer']

        def receive(report):
            with self.lock:
                self.analysis['report'] = deepcopy(report)
            if on_report is not None:
                on_report(report)

        verb = 'kata-analyze' if continuous else 'kata-search_analyze_cancellable'
        try:
            reply = self.engine.command(f'{verb} {player} 25 json true pvVisits true noResultValue true',
                                        on_report=receive, should_stop=cancelled.is_set)
            if reply.analysis is not None and (continuous or cancelled.is_set() or reply.body == 'play cancelled'):
                # The source continuous stream has no final snapshot on stop.
                # Once its response has drained, read the completed tree through
                # the native query rather than labeling the last periodic frame
                # as the final state of that search.
                receive(self.engine.analysis())
            with self.lock:
                if continuous:
                    if reply.body:
                        raise RuntimeError(f'Unexpected native analysis response: {reply.body}')
                    move = None
                else:
                    words = reply.body.split()
                    if len(words) != 2 or words[0] != 'play':
                        raise RuntimeError(f'Unexpected native search response: {reply.body}')
                    move = words[1]
                stopped = continuous or cancelled.is_set() or move == 'cancelled'
                self.analysis['status'] = 'stopped' if stopped else 'complete'
                if stopped:
                    move = None
                elif play:
                    # Commit and stop are ordered by the same lock. Cancellation
                    # accepted before this point must never play a partial move.
                    if move == 'resign':
                        self.resignation = dict(player=player, result=('W' if player == 'B' else 'B') + '+R',
                                                source='native GTP search response')
                    else:
                        self.play(move)
                self.analysis['move'] = move
                return dict(move=move, stopped=stopped)
        except BaseException:
            with self.lock:
                self.analysis['status'] = 'error'
            raise
        finally:
            with self.lock:
                self.cancelled = None

    def stop(self):
        with self.lock:
            if self.cancelled is not None:
                self.cancelled.set()
                self.engine.stop()

    def close(self):
        # Called by the serialized controller only after its current action has
        # drained; use stop() to interrupt an action running on another thread.
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
