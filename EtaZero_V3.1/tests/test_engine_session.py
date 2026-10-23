"""Real published CUDA models exercise shared JSON/GTP sessions and cancellation."""
import json
import os
from queue import Queue
import subprocess
from threading import Thread
import time

import pytest
from etazero.analysis import model_info
from etazero.config import ROOT, write_native
from etazero.engine_config import load_engine_config
from test_analysis_commands import model_and_profiles
from web.engine import Engine

pytestmark = pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='Host CUDA required')


@pytest.fixture(params=['puct', 'gumbel'])
def session_setup(tmp_path, model_and_profiles, request):
    model, source = model_and_profiles
    config = load_engine_config(source, environ={})
    config['analysis'].update(root_search_algo=request.param, nonroot_search_algo='puct',
                              reuse_tree=False, use_graph_search=False, search_threads=1,
                              nn_randomize=False, root_num_symmetries_to_sample=1,
                              use_lcb=True)
    _, info = model_info(model)
    path = tmp_path/'session.cfg'
    write_native({**config, 'network': {'canvas': info['canvas']}}, path)
    arguments = ['--config', str(path), '--model', str(model), '--model-id', info['id'],
                 '--device', config['analysis']['device'], '--seed', str(config['analysis']['seed'])]
    return model, config, arguments


def test_live_snapshots_stop_without_playing_and_reuse_model(session_setup):
    model, config, _ = session_setup
    with Engine(ROOT/'build/etazero', model, config) as engine:
        pid = engine.process.pid
        engine.command('new 5 hex')
        engine.command('play 0')
        snapshots = []

        def report(analysis):
            snapshots.append(analysis)
            if len(snapshots) == 2:
                engine.stop()

        result = engine.command({'command': 'genmove', 'visits': 100000, 'report_interval_ms': 1}, on_report=report)
        assert len(snapshots) >= 2
        assert result['cancelled'] and not result['played']
        assert result['state']['moves'] == [0]
        counts = [a['completed_visits'] for a in snapshots]
        assert counts == sorted(counts) and counts[-1] < 100000
        assert all(not a['final'] and a['board'] == result['analysis']['board'] for a in snapshots)
        final = engine.command('analyze 32')['analysis']
        assert final['completed_visits'] == 32 and final['final']
        assert engine.process.pid == pid
        for c in final['candidates']:
            assert 0 <= c['action'] < 25 and c['action'] != 0
            assert len(c['pv']) == len(set(c['pv'])) and 0 not in c['pv']
            if c['visits']:
                assert sum(c['wdl']) == pytest.approx(1)
                assert c['winrate'] == pytest.approx(c['wdl'][0] + .5*c['wdl'][1])
                assert c['win_loss'] == pytest.approx(c['wdl'][0] - c['wdl'][2])
            if final['root_search_algo'] == 'gumbel':
                assert c['lcb'] is None and c['raw_selection_weight'] is None
        if final['root_search_algo'] == 'puct':
            mass = sum(c['raw_selection_weight'] for c in final['candidates'])
            for c in final['candidates']:
                assert c['raw_selection_weight']/mass == pytest.approx(c['selection_weight'])
        assert engine.command('state')['state']['moves'] == [0]
        # Signal shutdown must wake an idle protocol reader without closing stdin.
        engine.process.terminate()
        assert engine.process.wait(timeout=5) == 130


def test_json_batch_positions_ids_and_recoverable_errors(session_setup):
    _, _, arguments = session_setup
    jobs = [dict(id='invalid', command='analyze', moves=[-1], size=5),
            dict(id='one', command='analyze', moves=[0, 5], size=5, rule='renju', visits=32, report_interval_ms=1),
            dict(id=2, command='analyze', moves=[1, 6], size=5, rule='hex', visits=16, report_interval_ms=0)]
    result = subprocess.run([str(ROOT/'build/etazero'), 'analysis', '--stream', 'true', '--protocol', 'json', *arguments],
                            input=''.join(json.dumps(job)+'\n' for job in jobs), text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert replies.pop(0)['event'] == 'ready'
    finals = {reply['id']: reply for reply in replies if reply.get('final')}
    assert set(finals) == {'invalid', 'one', 2} and not finals['invalid']['ok']
    assert finals['one']['state']['moves'] == [0, 5]
    assert finals[2]['state']['moves'] == [1, 6]
    assert finals['one']['analysis']['completed_visits'] == 32
    assert finals[2]['analysis']['completed_visits'] == 16
    assert any(reply['id'] == 'one' and not reply['final'] for reply in replies)


def test_gtp_framing_continuous_analysis_and_stop(session_setup):
    _, _, arguments = session_setup
    process = subprocess.Popen([str(ROOT/'build/etazero'), 'gtp', *arguments], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    lines = Queue()
    reader = Thread(target=lambda: [lines.put(line.rstrip('\n')) for line in process.stdout], daemon=True)
    reader.start()

    def send(command):
        process.stdin.write(command+'\n'); process.stdin.flush()

    def response(command, prefix):
        send(command); result = []
        while True:
            line = lines.get(timeout=30)
            if not line:
                if result: break
            else: result.append(line)
        assert result[0].startswith(prefix), result
        return result

    try:
        response('1 protocol_version', '=1 2')
        response('2 boardsize 5', '=2')
        response('3 etazero-rule hex', '=3')
        response('4 play b A1', '=4')
        response('5 play b B1', '?5')
        response('6 play w pass', '?6')
        send('7 kata-analyze w 1 maxmoves 3')
        assert lines.get(timeout=30) == '=7'
        live = lines.get(timeout=30)
        assert live.startswith('info move ') and 'winrate' in live and ' pv ' in live
        send('8 stop')
        while lines.get(timeout=30) != '=8': pass
        assert lines.get(timeout=30) == ''
        state = json.loads(response('9 showboard', '=9 ')[0][3:])
        assert state['moves'] == [20]
        response('10 genmove w', '=10 ')
        state = json.loads(response('11 showboard', '=11 ')[0][4:])
        assert len(state['moves']) == 2
        response('12 undo', '=12')
        response('13 kata-raw-nn 1', '?13')
        raw = json.loads(response('14 kata-raw-nn 0', '=14 ')[0].split(' ', 1)[1])
        assert len(raw['heads']) == 4 and len(raw['wdl']) == 3 and sum(raw['wdl']) == pytest.approx(1)
        send('15 kata-genmove_analyze w 1 visits 100000')
        assert lines.get(timeout=30) == '=15'
        assert lines.get(timeout=30).startswith('info move ')
        send('16 stop')
        cancelled = []
        while True:
            line = lines.get(timeout=30)
            if line == '=16': break
            cancelled.append(line)
        assert 'error search cancelled' in cancelled
        assert lines.get(timeout=30) == ''
        state = json.loads(response('17 showboard', '=17 ')[0].split(' ', 1)[1])
        assert state['moves'] == [20]
        response('18 quit', '=18')
        process.wait(timeout=10)
        assert process.returncode == 0, process.stderr.read()
    finally:
        if process.poll() is None: process.kill()
        process.wait(timeout=10)
        process.stdin.close(); process.stdout.close(); process.stderr.close()
