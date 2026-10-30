"""Outer match schedule persistence and recovery, using native pairing proposals.

Native match holds the results-directory lock throughout these requests and
actual game writes. No model, search, or pairing RNG is implemented here.
"""
from collections import Counter
import json
from pathlib import Path

from .run import save_json

STATE_PREFIX = 'ETAZERO_MATCH_STATE '
READY_MARKER = 'ETAZERO_MATCH_READY'


def read_game(path, index, pair):
    game = json.loads(path.read_text())
    if (game['game_index'] != index or [game['black'], game['white']] != pair or
            game['outcome'] not in ('B', 'W', 'D', 'NR')):
        raise ValueError(f'Game does not match its committed schedule: {path}')
    return game


class MatchCache:
    def __init__(self, request):
        self.directory = Path(request['directory'])
        self.config = request['resolved_config']
        self.extension = request['extension_config']
        self.cwd = request['working_directory']
        self.count = request['num_games']
        self.per_pair = request['games_per_pair']
        self.plan = None
        self.completed = []
        path = self.directory / 'plan.json'
        if path.exists():
            self.plan = json.loads(path.read_text())
            key, expected = ('extension_config', self.extension) if self.extension else ('resolved_config', self.config)
            if self.plan.get(key) != expected:
                raise ValueError('Match configuration differs from the saved schedule; use a new results directory')
            if self.plan['working_directory'] != self.cwd:
                raise ValueError('Match working directory changed; relative resource paths must retain their meaning')
            if not self.plan['matchups']:
                raise ValueError('Saved match schedule is empty')
            for index, pair in enumerate(self.plan['matchups']):
                game = self.directory / f'game-{index}.json'
                if game.exists():
                    read_game(game, index, pair)
                    self.completed.append(index)
        elif any(p.name != '.native.lock' and not p.name.startswith('plan.json.tmp.')
                 for p in self.directory.iterdir()):
            raise ValueError('A new match requires an empty results directory')

    def state(self):
        generate = self.plan is None or bool(self.extension and len(self.plan['matchups']) < self.count)
        if self.plan is not None and not self.extension and len(self.plan['matchups']) != self.count:
            raise ValueError('Saved match schedule length differs from numGamesTotal')
        # New schedules consume the native pairer's complete requested rounds.
        # Extension only draws the deficit after completing an odd saved round.
        if self.plan is None:
            proposal_games = None
        elif generate:
            saved = len(self.plan['matchups'])
            proposal_games = max(0, self.count - saved - saved % 2)
        else:
            proposal_games = 0
        return dict(plan=self.plan, completed=self.completed, generate=generate, proposal_games=proposal_games)

    def schedule(self, proposed):
        if not self.state()['generate']:
            raise ValueError('Match schedule already exists; no new pairing proposal was requested')
        if self.plan is None:
            plan = dict(proposed, resolved_config=self.config, working_directory=self.cwd)
            if self.extension:
                plan['extension_config'] = self.extension
            if self.per_pair > 0:
                counts, matchups = Counter(), []
                for pair in proposed['matchups']:
                    key = tuple(sorted(pair))
                    if counts[key] < self.per_pair:
                        matchups.append(pair)
                        counts[key] += 1
                plan['matchups'] = matchups
        else:
            matchups = list(self.plan['matchups'])
            # Complete the saved odd color round before drawing fresh native
            # rounds. The prior prefix and its game IDs are never rewritten.
            if len(matchups) % 2:
                matchups.append(list(reversed(matchups[-1])))
            matchups.extend(proposed['matchups'][:self.count - len(matchups)])
            plan = dict(self.plan, matchups=matchups, resolved_config=self.config)
        if len(plan['matchups']) != self.count:
            raise ValueError('Native pairing proposal does not satisfy the requested game count')
        if self.plan is not None:
            previous = self.directory / 'plans' / f'{len(self.plan["matchups"])}.json'
            previous.parent.mkdir(exist_ok=True)
            if not previous.exists():
                save_json(previous, self.plan)
        save_json(self.directory / 'plan.json', plan)
        self.plan = plan
        return self.state()
