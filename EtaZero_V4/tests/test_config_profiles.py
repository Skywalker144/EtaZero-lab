"""Outer directory profiles preserve native names and resolve actual launch inputs."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from etazero import ROOT
from etazero.config import RunConfig, load_config
from etazero.experiment import experiment_plan
from etazero.profiles import default_output, load_run_profile, native_arguments


def write_profile(path, config=None):
    path.mkdir(parents=True, exist_ok=True)
    if config is None:
        (path / 'selfplay.cfg').write_text('maxVisits = 13\n')
        config = RunConfig(selfplay_config=str(path / 'selfplay.cfg'), canvas=5,
            batch_size=4, games_per_round=4, samples_per_epoch=8, model_kind='b2c16-fson-mish')
    (path / 'run.cfg').write_text(
        '[run]\n'
        f'selfplay_config = {shlex.quote(config.selfplay_config)}\n'
        f'game = {config.game}\nalgorithm = {config.algorithm}\n'
        f'games_per_round = {config.games_per_round}\n'
        f'publish_weights = {config.publish_weights}\nauto_plot = false\niterations = 2\n')
    def extra_args(args):
        lines = []
        for arg in args:
            if arg.startswith('-'):
                lines.append(arg[1:])
            else:
                lines[-1] += ' = ' + shlex.quote(arg)
        return '\n'.join(lines)
    network = f'model-kind = {config.model_kind}'
    if config.is_muzero:
        (path / 'network.json').write_text(json.dumps(config.muzero_network))
        network = 'muzero-config = network.json'
    (path / 'train.cfg').write_text(
        f'[train]\n{network}\npos-len = {config.canvas}\nbatch-size = {config.batch_size}\n'
        f'samples-per-epoch = {config.samples_per_epoch}\nmax-epochs-this-instance = {config.epochs_per_round}\n'
        + extra_args(config.train_args) + '\n')
    (path / 'shuffle.cfg').write_text(
        f'[shuffle]\nnum-processes = {config.shuffle_processes}\n'
        f'approx-rows-per-out-file = {config.shuffle_rows_per_file}\nkeep-target-rows = {config.shuffle_keep_rows}\n'
        + (f'muzero-unroll-steps = {config.muzero_unroll_steps}\n' if config.is_muzero else '')
        + extra_args(config.shuffle_args) + '\n')
    return replace(config, auto_plot=False).validate()


def shell(script, *args, env=None):
    return subprocess.run(['bash', str(ROOT / 'scripts' / script), *map(str, args)],
                          cwd='/tmp', env={**os.environ, **(env or {})}, text=True, capture_output=True)


def test_native_keys_values_and_local_overrides(tmp_path):
    path = tmp_path / 'profile with spaces'
    expected = write_profile(path)
    assert load_config(path).serializable() == expected.serializable()
    (path / 'train.cfg.local').write_text(
        '[train]\nbatch-size = 8\nlookahead-k = 11\nno-compile\n'
        'lr-schedule = "(0,2.5),(20M,0.5)"\nexportprefix = "100% # literal" # comment\n')
    args = native_arguments(path / 'train.cfg', 'train')
    assert args[-5:] == ['-no-compile', '-lr-schedule', '(0,2.5),(20M,0.5)', '-exportprefix', '100% # literal']
    # Output ownership cannot be bypassed with the new file format.
    with pytest.raises(ValueError, match='owned by the outer controller'):
        load_config(path)
    (path / 'train.cfg.local').write_text('[train]\nbatch-size = 8\nlookahead-k = 11\nno-compile\n')
    actual = load_config(path)
    assert actual.batch_size == 8
    assert actual.train_args == ['-lookahead-k', '11', '-no-compile']
    assert (path / 'selfplay.cfg').read_text() == 'maxVisits = 13\n'


@pytest.mark.parametrize('change', ['[run]\nmaxVisits = 1\n', '[run]\nauto_plot = perhaps\n',
                                   '[DEFAULT]\nalgorithm = alphazero\n', '[network]\nchannels = 16\n'])
def test_outer_profile_rejects_unowned_or_mistyped_settings(tmp_path, change):
    write_profile(tmp_path)
    (tmp_path / 'run.cfg.local').write_text(change)
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_shared_native_option_files_and_muzero_network_paths(tmp_path):
    from test_controller_recovery import recovery_config
    shared = tmp_path / 'shared'; shared.mkdir()
    expected = write_profile(shared, recovery_config(shared, 'gumbel_muzero'))
    arm = tmp_path / 'arm'; arm.mkdir()
    (arm / 'run.cfg').write_text((shared / 'run.cfg').read_text() +
                                'train_config = ../shared/train.cfg\nshuffle_config = ../shared/shuffle.cfg\n')
    assert load_config(arm).serializable() == expected.serializable()
    # Exact native option names: abbreviations must not replace boundary ownership.
    (shared / 'train.cfg.local').write_text('[train]\nmax-epochs-this = 99\n')
    with pytest.raises(ValueError, match='owned by the outer controller'):
        load_config(arm)


def test_script_launch_defaults_cli_precedence_and_no_dry_run_writes(tmp_path):
    directory = tmp_path / 'profile'; write_profile(directory)
    (directory / 'run.cfg.local').write_text('[run]\niterations = 3\nrun_dir = "../saved run"\n')
    result = shell('run.sh', '--dry-run', '--iterations', '4', env={'CONFIG_DIR': str(directory)})
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['iterations'] == 4
    assert plan['run_dir'] == str(tmp_path / 'saved run')
    assert not Path(plan['run_dir']).exists()
    assert shell('run.sh', '--version').returncode == 0
    check = shell('run.sh', 'check-config', '--config-dir', directory)
    assert check.returncode == 0, check.stderr
    assert json.loads(check.stdout)['canvas'] == 5
    invalid = shell('run.sh', '--config-dir', directory, '--iterations', '0', '--dry-run')
    assert invalid.returncode != 0 and 'positive cumulative' in invalid.stderr
    assert default_output(ROOT / 'configs' / 'any-arm' / 'run.cfg', 'run.cfg') == ROOT / 'data' / 'any-arm'
    with pytest.raises(ValueError, match='explicit output'):
        default_output(directory, 'run.cfg')


def test_autoexp_discovers_arms_and_uses_native_profiles(tmp_path):
    directory = tmp_path / 'experiment'; directory.mkdir()
    for name in ('z', 'a space'):
        write_profile(directory / name)
    (directory / 'not-an-arm').mkdir()
    (directory / 'exp.cfg').write_text('[experiment]\niterations = 4\ngpus = 0,1\nshared_init = false\n')
    output = tmp_path / 'output'
    result = shell('autoexp.sh', '--config-dir', directory, '--work-dir', output, '--dry-run')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert [arm['name'] for arm in plan['arms']] == ['a space', 'z']
    assert plan['slots'] == ['0', '1'] and plan['iterations'] == 4
    assert not output.exists()
    (directory / 'match.cfg').write_text('maxVisits = 19\n')
    (directory / 'exp.cfg.local').write_text(
        '[arms]\nchosen = "a space"\n[evaluation]\nconfig = match.cfg\ngames_per_pair = 3\nfinal_cross = true\n')
    plan = experiment_plan(directory, output)
    assert [arm['name'] for arm in plan['arms']] == ['chosen']
    assert plan['evaluation'] == dict(config=str(directory / 'match.cfg'), games_per_pair=3, final_cross=True)


@pytest.mark.skipif(os.environ.get('ETAZERO_GPU_TESTS') != '1', reason='requires native CUDA selfplay and learner')
@pytest.mark.parametrize('algorithm', ['alphazero', 'gumbel_muzero'])
def test_directory_cli_real_training_and_resume(tmp_path, algorithm):
    from test_controller_recovery import recovery_config
    directory = tmp_path / 'profile'; directory.mkdir()
    write_profile(directory, recovery_config(directory, algorithm))
    output = tmp_path / 'output'
    command = ('--config-dir', directory, '--run-dir', output)
    first = shell('run.sh', *command)
    assert first.returncode == 0, first.stdout[-6000:] + first.stderr[-6000:]
    state = json.loads((output / 'state.json').read_text())
    assert state['completed_rounds'] == 2 and state['accepted_model']
    commits = {path: path.read_bytes() for path in (output / 'rounds').glob('*/commit.json')}
    second = shell('run.sh', *command)
    assert second.returncode == 0, second.stderr
    assert json.loads((output / 'state.json').read_text()) == state
    assert all(path.read_bytes() == raw for path, raw in commits.items())
