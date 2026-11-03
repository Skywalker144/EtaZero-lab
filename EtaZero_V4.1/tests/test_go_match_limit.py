"""Move-limit adjudication uses actual Go scoring and remains recoverable."""
from collections import Counter
import json
import os
import re
import signal
import subprocess

import pytest

from etazero import ROOT, autoelo
from etazero.evaluation import load_games, run_match, summarize_match, summarize_endings
from etazero.match_cache import MatchCache
from test_match import match_config, export_models, gpu
from test_match_cache import request, proposal
from test_arena import history
from test_controller_recovery import wait_until
from match_dispatch_worker import worker_command


def go_config(path, *, scoring='AREA', komi=7.5, moves=0, models=None):
    config = match_config(path, models or [('/dev/null', False)] * 2, game='go', games=2, moves=moves)
    config.write_text(config.read_text().replace('komiMean = 0\n', f'komiMean = {komi}\n'))
    with config.open('a') as stream:
        stream.write(f'koRules = POSITIONAL\nscoringRules = {scoring}\ntaxRules = NONE\n'
                     'hasButtons = false\nmultiStoneSuicideLegals = false\n'
                     f'debugSkipNeuralNet = {str(models is None).lower()}\n')
    return config


@pytest.mark.parametrize('scoring', ['AREA', 'TERRITORY'])
@pytest.mark.parametrize('komi,outcome,result', [(7.5, 'W', 'W+7.5'), (-7.5, 'B', 'B+7.5'), (0, 'D', '0')])
def test_empty_board_limit_scores_komi_and_retains_sgf(tmp_path, scoring, komi, outcome, result):
    config = go_config(tmp_path / 'match.cfg', scoring=scoring, komi=komi)
    snapshot = tmp_path / 'source.tar.gz'; snapshot.touch()
    output = tmp_path / 'evaluation'
    ratings = run_match(output, config, source_snapshot=snapshot)
    plan, games = load_games(output / 'results')
    assert plan['go_move_limit_scoring'] is True
    assert [g['outcome'] for g in games] == [outcome] * 2
    assert all(g['moves'] == 0 and g['hit_turn_limit'] and g['end_reason'] == 'move_limit' for g in games)
    assert all(f'RE[{result}]' in g['sgf'] and 'endReason=move_limit' in g['sgf'] for g in games)
    assert ratings['games_truncated'] == 2 and ratings['truncation_fraction'] == 1
    assert json.loads((output / 'results/summary.json').read_text())['truncation_fraction'] == 1
    saved = {p: p.read_bytes() for p in (output / 'results').glob('game-*.json')}
    assert run_match(output, config, env=dict(os.environ, CUDA_VISIBLE_DEVICES='')) == ratings
    assert all(p.read_bytes() == raw for p, raw in saved.items())


def test_partial_summary_denominator_and_unknown_endings(tmp_path):
    plan = proposal([[0, 1], [1, 0], [0, 1], [1, 0]])
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    for index, reason in [(0, 'move_limit'), (2, 'resignation')]:
        (tmp_path / f'game-{index}.json').write_text(json.dumps(dict(
            game_index=index, black=0, white=1, outcome='W', end_reason=reason)))
    summary = summarize_match(tmp_path)
    assert not summary['completed'] and summary['games_completed'] == 2
    assert summary['games_truncated'] == 1 and summary['truncation_fraction'] == .5
    assert summary['end_reasons'] == {'move_limit': 1, 'resignation': 1}
    assert summarize_endings(Counter())['truncation_fraction'] is None
    assert summarize_endings(Counter(unknown=1, move_limit=1))['truncation_fraction'] is None


def test_old_schedule_cannot_resume_under_new_go_adjudication(tmp_path):
    inputs = request(tmp_path / 'results', 2)
    MatchCache(inputs).schedule(proposal([[0, 1], [1, 0]]))
    old = (tmp_path / 'results/plan.json').read_bytes()
    with pytest.raises(ValueError, match='Go move-limit scoring differs'):
        MatchCache(dict(inputs, go_move_limit_scoring=True))
    assert (tmp_path / 'results/plan.json').read_bytes() == old


def test_autoelo_new_cache_identity_and_batch_truncation_statistics(tmp_path):
    data = tmp_path / 'runs'
    history(data / 'a', [5, 10], game='go')
    config = go_config(tmp_path / 'match.cfg')
    output = tmp_path / 'elo'
    plan = autoelo.autoelo_plan(output, data, config, games_per_pair=2)
    for pair in plan['pairs']:
        players = [p for name in pair['players'] for p in plan['players'] if p['id'] == name]
        resolved = autoelo.resolved_pair(config, autoelo.match_overlay(dict(
            players=players, pairs=[pair['players']], games_per_pair=2)), ROOT / 'build/katago')
        assert resolved.pop('go_move_limit_scoring') is True
        old_key = autoelo.identity({k: v for k, v in resolved.items() if k != 'resolved_config'})
        assert old_key != pair['cache_key']
    ratings = autoelo.run_autoelo(output, config, data=data, games_per_pair=2)
    assert ratings['games_truncated'] == ratings['games'] > 0
    assert ratings['truncation_fraction'] == 1
    session = (output / 'latest').resolve()
    assert autoelo.run_autoelo(session, fit_only=True, binary='/missing') == ratings
    saved = {p: p.read_bytes() for p in (output / 'pair_cache').glob('*/match/results/game-*.json')}
    assert autoelo.run_autoelo(output, config, data=data, games_per_pair=2) == ratings
    assert all(p.read_bytes() == raw for p, raw in saved.items())


@gpu
def test_cuda_mixed_go_limit_scores_final_board_and_plain_sgf(tmp_path):
    ordinary, muzero = export_models(tmp_path, 'go')
    config = go_config(tmp_path / 'match.cfg', komi=100, moves=1,
                       models=[(ordinary, False), (muzero, True)])
    output = tmp_path / 'evaluation'
    ratings = run_match(output, config, max_loaded_models=2)
    _, games = load_games(output / 'results')
    assert ratings['games_truncated'] == 2
    for game in games:
        assert game['moves'] == 1 and game['end_reason'] == 'move_limit' and game['outcome'] == 'W'
        move = re.search(r';B\[([^]]*)\]', game['sgf']).group(1)
        # On an empty 3x3 board, a single black stone claims all 9 points;
        # a pass leaves 0 points. Neither position has naturally ended.
        assert f'RE[W+{91 if move else 100}]' in game['sgf']
        assert 'endReason=move_limit,adjudication=training_score' in game['sgf']
    plain = tmp_path / 'plain'
    process = subprocess.run([str(ROOT / 'build/katago'), 'match', '-config', str(config),
                              '-sgf-output-dir', str(plain)], capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert sum(p.read_text().count('endReason=move_limit') for p in plain.glob('*.sgfs')) == 2

    # Interrupt an actual Go search. This must leave an unfinished scheduled
    # game, not pass the partial board through the move-limit scorer.
    config.write_text(config.read_text().replace('maxMovesPerGame = 1\n', 'maxMovesPerGame = 100000\n')
                      .replace('maxVisits = 8\n', 'maxVisits = 2048\n')
                      .replace('numGameThreads = 2\n', 'numGameThreads = 1\n') + 'logMoves = true\n')
    interrupted = tmp_path / 'interrupted'
    native_log = tmp_path / 'interrupt.native.log'
    command = [str(ROOT / 'build/katago'), 'match', '-config', str(config),
               '-results-dir', str(interrupted), '-log-file', str(native_log)]
    with (tmp_path / 'interrupt.log').open('w') as log:
        process = subprocess.Popen(worker_command(command), stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_until(lambda: native_log.exists() and 'Move 0 made:' in native_log.read_text(), seconds=60)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=30) == 0
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
    assert not list(interrupted.glob('game-*.json'))
