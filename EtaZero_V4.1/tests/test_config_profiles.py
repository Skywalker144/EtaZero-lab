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
from etazero.profiles import default_output, load_run_profile, native_arguments, read_sections


def write_profile(path, config=None):
    path.mkdir(parents=True, exist_ok=True)
    if config is None:
        (path / 'selfplay.cfg').write_text('maxVisits = 13\n')
        config = RunConfig(selfplay_config=str(path / 'selfplay.cfg'), canvas=5,
            batch_size=4, bootstrap_games=4, train_steps=2, model_kind='b2c16-fson-mish')
    (path / 'run.cfg').write_text(
        '[run]\n'
        f'selfplay_config = {shlex.quote(config.selfplay_config)}\n'
        f'game = {config.game}\nalgorithm = {config.algorithm}\n'
        f'bootstrap_games = {config.bootstrap_games}\ntrain_steps = {config.train_steps}\n'
        f'replay_ratio = {config.replay_ratio}\nbootstrap_game_multiplier = {config.bootstrap_game_multiplier}\n'
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
        + extra_args(config.train_args) + '\n')
    (path / 'shuffle.cfg').write_text(
        f'[shuffle]\nnum-processes = {config.shuffle_processes}\nmin-rows = {config.min_rows}\n'
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
    invalid = shell('run.sh', '--config-dir', directory, '--iterations', '-1', '--dry-run')
    assert invalid.returncode != 0 and 'nonnegative final iteration' in invalid.stderr
    assert default_output(ROOT / 'configs' / 'any-arm' / 'run.cfg', 'run.cfg') == ROOT / 'data' / 'any-arm'
    with pytest.raises(ValueError, match='explicit output'):
        default_output(directory, 'run.cfg')


def test_script_launch_without_limits_defaults_to_unlimited(tmp_path):
    directory = tmp_path / 'profile'; write_profile(directory)
    profile = directory / 'run.cfg'
    profile.write_text(profile.read_text().replace('iterations = 2\n', ''))
    output = tmp_path / 'output'
    result = shell('run.sh', '--dry-run', '--run-dir', output,
                   env={'CONFIG_DIR': str(directory)})
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['iterations'] is None and plan['max_hours'] is None
    assert not output.exists()


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


def test_include_order_local_overrides_and_removing_native_flags(tmp_path):
    (tmp_path / 'common.cfg').write_text('[train]\nbatch-size = 8\nno-compile\n')
    (tmp_path / 'common.cfg.local').write_text('[train]\nbatch-size = 12\n')
    profile = tmp_path / 'train.cfg'
    profile.write_text('[train]\nbatch-size = 4\n@include = "common.cfg" # source order\nlookahead-k = 3\n')
    assert native_arguments(profile, 'train') == ['-batch-size', '12', '-no-compile', '-lookahead-k', '3']
    profile.with_name('train.cfg.local').write_text('[train]\nbatch-size = 16\nno-compile = # remove inherited flag\n')
    assert native_arguments(profile, 'train') == ['-batch-size', '16', '-lookahead-k', '3']
    # The include can override a parent key, but cannot hide a duplicate written
    # twice in that parent file (the same rule as native ConfigParser).
    profile.write_text(profile.read_text() + 'batch-size = 20\n')
    with pytest.raises(ValueError, match='duplicate key'):
        read_sections(profile, ['train'])


def test_nested_include_paths_belong_to_declaring_files(tmp_path):
    from test_controller_recovery import recovery_config
    shared = tmp_path / 'shared'; shared.mkdir()
    expected = write_profile(shared, recovery_config(shared, 'muzero'))
    (shared / 'run.cfg').write_text((shared / 'run.cfg').read_text() +
                                  'train_config = train.cfg\nshuffle_config = shuffle.cfg\nrun_dir = output\n')
    arm = tmp_path / 'arms' / 'one'; arm.mkdir(parents=True)
    (arm / 'run.cfg').write_text('@include ../../shared/run.cfg\n[run]\ntrain_config = train.cfg\n')
    (arm / 'train.cfg').write_text('@include ../../shared/train.cfg\n')
    actual, launch = load_run_profile(arm)
    assert actual.serializable() == expected.serializable()
    assert launch['run_dir'] == str(shared / 'output')
    # No explicit resource paths: defaults belong to the selected arm itself.
    (shared / 'run.cfg').write_text('[run]\nbootstrap_games = 4\ntrain_steps = 2\n')
    (arm / 'run.cfg').write_text('@include ../../shared/run.cfg\n[run]\nalgorithm = muzero\ngame = renju\n')
    (arm / 'shuffle.cfg').write_bytes((shared / 'shuffle.cfg').read_bytes())
    (arm / 'selfplay.cfg').write_text('maxVisits = 3\n')
    assert load_config(arm).selfplay_config == str(arm / 'selfplay.cfg')


def test_experiment_include_resolves_arm_and_evaluation_paths(tmp_path):
    shared = tmp_path / 'shared'; shared.mkdir()
    write_profile(shared / 'arm')
    (shared / 'match.cfg').write_text('maxVisits = 7\n')
    (shared / 'exp.cfg').write_text('[experiment]\niterations = 2\n[arms]\nchosen = arm\n'
                                  '[evaluation]\nconfig = match.cfg\ngames_per_pair = 2\n')
    selected = tmp_path / 'selected'; selected.mkdir()
    (selected / 'exp.cfg').write_text('@include ../shared/exp.cfg\n[experiment]\niterations = 3\n')
    plan = experiment_plan(selected, tmp_path / 'output')
    assert plan['iterations'] == 3
    assert plan['arms'][0]['config_path'] == str(shared / 'arm')
    assert plan['evaluation']['config'] == str(shared / 'match.cfg')


def test_include_errors_name_the_input_boundary(tmp_path):
    profile = tmp_path / 'run.cfg'
    profile.write_text('@include missing.cfg\n')
    with pytest.raises(FileNotFoundError, match='missing.cfg'):
        read_sections(profile, ['run'])
    profile.write_text('@include child.cfg\n')
    (tmp_path / 'child.cfg').write_text('@include run.cfg\n')
    with pytest.raises(ValueError, match='include cycle'):
        read_sections(profile, ['run'])
    profile.write_text('@unknown child.cfg\n')
    with pytest.raises(ValueError, match='expected @include'):
        read_sections(profile, ['run'])


@pytest.mark.skipif(os.environ.get('ETAZERO_EIGEN_TESTS') != '1', reason='requires native Eigen build')
@pytest.mark.parametrize('algorithm', ['alphazero', 'gumbel_muzero'])
def test_included_profile_real_cpu_training_and_resume(tmp_path, algorithm):
    from test_controller_recovery import recovery_config
    shared = tmp_path / 'shared'; shared.mkdir()
    config = recovery_config(shared, algorithm)
    config = replace(config, train_steps=3,
                     train_args=tuple(a for a in config.train_args if a != '-use-fp16'))
    with Path(config.selfplay_config).open('a') as out:
        out.write('inputsUseNHWC = true\neigenUseFP16 = false\n')
    write_profile(shared, config)
    (shared / 'run.cfg').write_text((shared / 'run.cfg').read_text() +
                                  'train_config = train.cfg\nshuffle_config = shuffle.cfg\n')
    arm = tmp_path / 'arm'; arm.mkdir()
    (arm / 'run.cfg').write_text('@include ../shared/run.cfg\n[run]\ntrain_config = train.cfg\n')
    (arm / 'train.cfg').write_text('@include ../shared/train.cfg\n')
    output = tmp_path / 'output'
    command = ('--config-dir', arm, '--run-dir', output, '--binary', ROOT / 'build-eigen/katago')
    env = dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = shell('run.sh', *command, env=env)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    state = json.loads((output / 'state.json').read_text())
    assert state['completed_rounds'] == 2 and state['total_samples'] > 0
    assert json.loads((output / 'rounds/000002/result.json').read_text())['input_model'].startswith('round-000001-s')
    committed = (output / 'train/checkpoint.ckpt').read_bytes()
    result = shell('run.sh', *command, env=env)
    assert result.returncode == 0, result.stderr
    assert (output / 'train/checkpoint.ckpt').read_bytes() == committed


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


def test_display_groups_share_native_namespace_and_include_order(tmp_path):
    common = tmp_path / 'common.cfg'
    common.write_text('[inference]\nnnMaxBatchSize = 16\n[writer]\nmaxRowsPerTrainFile = 100\n')
    path = tmp_path / 'selfplay.cfg'
    path.write_text('@include common.cfg\n[selfplay]\nnnMaxBatchSize = 8\n')
    path.with_name('selfplay.cfg.local').write_text('[inference]\nnnMaxBatchSize = 4\n')
    assert read_sections(path, ['selfplay']) == {
        'selfplay': {'nnMaxBatchSize': '4', 'maxRowsPerTrainFile': '100'}}
    # Display groups must not hide a duplicate native key within one file.
    path.write_text('[selfplay]\nnnMaxBatchSize = 8\n[inference]\nnnMaxBatchSize = 4\n')
    with pytest.raises(ValueError, match='duplicate key'):
        read_sections(path, ['selfplay'])


def test_grouped_cli_and_gumbel_activation_namespace(tmp_path):
    path = tmp_path / 'train.cfg'
    path.write_text('[training]\nbatch-size = 4\n[optimizer]\nuse-adamw\n[precision]\nno-compile =\n')
    assert native_arguments(path, 'train') == ['-batch-size', '4', '-use-adamw']
    path.write_text('[puct]\ncpuctExploration = 1.1\n[gumbel]\ngumbelMaxConsideredActions = 8\n')
    assert read_sections(path, ['selfplay', 'gumbel']) == {
        'selfplay': {'cpuctExploration': '1.1'}, 'gumbel': {'gumbelMaxConsideredActions': '8'}}
