"""Output paths follow the selected config, independent of inheritance and launch cwd."""
from config_samples import CONFIGS
from pathlib import Path
import shutil

import pytest

from etazero.config import ROOT, fingerprint, load_config
from etazero.experiment import experiment_plan, write_arm_config


@pytest.fixture
def version(tmp_path,monkeypatch):
    shutil.copytree(CONFIGS,tmp_path/'configs')
    monkeypatch.setattr('etazero.config.ROOT',tmp_path)
    return tmp_path


@pytest.mark.parametrize('name',[
    'baseline','minimal_test','smoke_test','muzero','raw_muzero',
    'sweep/arm_a','sweep/arm_b','autoexp_example/arm_a',
])
def test_default_output_mirrors_full_config_path(version,name,monkeypatch):
    caller=version/'caller';caller.mkdir();monkeypatch.chdir(caller)
    config=load_config(version/'configs'/name)
    assert config['run']['run_dir']==str(version/'data'/name)
    assert not (version/'data').exists() and not (caller/'data').exists()


def test_parent_output_is_not_inherited(version):
    baseline=version/'configs/baseline/run.cfg'
    baseline.write_text(baseline.read_text().replace('[run]','[run]\nrun_dir=data/parent',1))
    (baseline.parent/'run.cfg.local').write_text('[run]\nrun_dir=data/parent_local\n')
    assert load_config(baseline.parent)['run']['run_dir']==str(version/'data/parent_local')
    parent=version/'configs/muzero/run.cfg'
    parent.write_text(parent.read_text().replace('[run]','[run]\nrun_dir=data/muzero_custom',1))
    assert load_config(version/'configs/smoke_test')['run']['run_dir']==str(version/'data/smoke_test')
    assert load_config(version/'configs/raw_muzero')['run']['run_dir']==str(version/'data/raw_muzero')


def test_config_local_and_cli_output_precedence(version,monkeypatch):
    directory=version/'configs/smoke_test'
    run=directory/'run.cfg'
    run.write_text(run.read_text().replace('[run]','[run]\nrun_dir=data/custom',1))
    assert load_config(directory)['run']['run_dir']==str(version/'data/custom')
    local=directory/'run.cfg.local';local.write_text('[run]\nrun_dir=data/local\n')
    assert load_config(directory)['run']['run_dir']==str(version/'data/local')
    caller=version/'caller';caller.mkdir();monkeypatch.chdir(caller)
    assert load_config(directory,run_dir='output')['run']['run_dir']==str(caller/'output')
    local.write_text('[run]\nrun_dir=\n')
    automatic=load_config(directory)
    assert automatic['run']['run_dir']==str(version/'data/smoke_test')
    assert fingerprint(load_config(directory,run_dir=version/'data/smoke_test'))==fingerprint(automatic)
    with pytest.raises(ValueError,match='must not be empty'):
        load_config(directory,run_dir='')


def test_external_config_requires_its_own_output(version):
    directory=version/'external';directory.mkdir()
    run=directory/'run.cfg';run.write_text('[run]\nextends=smoke_test\n')
    with pytest.raises(ValueError,match='requires an explicit run_dir'):
        load_config(directory)
    output=version/'external_data'
    assert load_config(directory,run_dir=output)['run']['run_dir']==str(output)
    run.write_text('[run]\nextends=smoke_test\nrun_dir=data/external\n')
    assert load_config(directory)['run']['run_dir']==str(version/'data/external')


def test_symlinks_are_resolved_before_arm_overlap_checks(version):
    (version/'data').mkdir()
    (version/'data/shared').mkdir()
    (version/'data/alias').symlink_to(version/'data/shared',target_is_directory=True)
    directory=version/'configs/autoexp_example'
    for name,output in [('arm_a','shared'),('arm_b','alias')]:
        (directory/name/'run.cfg').write_text(f'[run]\nextends=smoke_test\nrun_dir=data/{output}\n')
    with pytest.raises(ValueError,match='non-overlapping'):
        experiment_plan(directory,environ={},work_dir=version/'work')


def test_experiment_arms_match_standalone_paths_and_survive_relocation(version):
    directory=version/'configs/sweep'
    plan=experiment_plan(directory,environ={},work_dir=version/'work')
    assert [arm['name'] for arm in plan['arms']] == ['arm_a', 'arm_b']
    for arm in plan['arms']:
        standalone=load_config(Path(arm['config_dir']))
        assert arm['run_dir']==str(version/'data/sweep'/arm['name'])
        assert arm['config']==standalone
        resolved=version/'generated'/arm['name']
        write_arm_config(arm['config'],resolved)
        assert load_config(resolved)==standalone
    assert not (version/'work').exists() and not (version/'data').exists()


def derived(version, name, parent, overrides=''):
    directory = version/'configs'/name
    directory.mkdir(parents=True)
    (directory/'run.cfg').write_text(f'[run]\nextends={parent}\n{overrides}')
    return directory


def test_qualified_parents_start_at_configs_and_preserve_child_output(version):
    parent = derived(version, 'family/base', 'baseline', 'seed=19\nrun_dir=data/parent\n')
    (parent/'run.cfg.local').write_text('[run]\nseed=99\n')
    # A same-named path beneath the child's siblings cannot shadow a qualified path.
    derived(version, 'group/family/base', 'baseline', 'seed=23\n')
    child = derived(version, 'group/child', 'family/base')
    result = load_config(child)
    assert result['run']['seed'] == 19
    assert result['run']['run_dir'] == str(version/'data/group/child')
    assert fingerprint(result) == fingerprint(load_config(child, run_dir=version/'data/group/child'))
    (child/'run.cfg.local').write_text('[run]\nseed=31\n')
    assert load_config(child)['run']['seed'] == 31


def test_bare_parent_names_resolve_beside_each_declaring_config(version):
    # Both levels contain a "base"; a nested parent must use its own sibling.
    derived(version, 'selected/base', 'baseline', 'seed=23\n')
    derived(version, 'other/base', 'baseline', 'seed=19\n')
    derived(version, 'other/middle', 'base')
    child = derived(version, 'selected/child', 'other/middle')
    assert load_config(child)['run']['seed'] == 19


def test_qualified_parent_is_independent_of_launch_directory(version, monkeypatch):
    derived(version, 'nested/parent', 'baseline', 'seed=19\n')
    child = derived(version, 'nested/child', 'nested/parent')
    caller = version/'caller'
    caller.mkdir()
    monkeypatch.chdir(caller)
    assert load_config(child)['run']['seed'] == 19


def test_cycles_across_qualified_and_bare_parents_are_rejected(version):
    child = derived(version, 'family/child', 'other/parent')
    derived(version, 'other/parent', 'sibling')
    derived(version, 'other/sibling', 'family/child')
    with pytest.raises(ValueError, match='inheritance cycle'):
        load_config(child)


@pytest.mark.parametrize('parent', ['/tmp/config', '../baseline', 'family/../baseline', './baseline', '.'])
def test_parent_paths_reject_absolute_and_dot_components(version, parent):
    child = derived(version, 'child', parent)
    with pytest.raises(ValueError, match='extends must name'):
        load_config(child)


def test_missing_qualified_parent_is_diagnostic(version):
    child = derived(version, 'child', 'missing/parent')
    with pytest.raises(ValueError, match='Missing configuration directory/run.cfg:.*missing/parent'):
        load_config(child)


def test_qualified_parent_cannot_escape_configs_through_symlink(version):
    outside = version/'outside'
    shutil.copytree(version/'configs/baseline', outside)
    (version/'configs/alias').symlink_to(outside, target_is_directory=True)
    child = derived(version, 'child', 'alias/')
    with pytest.raises(ValueError, match='must stay under configs'):
        load_config(child)
