"""Output paths follow the selected config, independent of inheritance and launch cwd."""
from pathlib import Path
import shutil

import pytest

from etazero.config import ROOT, fingerprint, load_config
from etazero.experiment import experiment_plan, write_arm_config


@pytest.fixture
def version(tmp_path,monkeypatch):
    shutil.copytree(ROOT/'configs',tmp_path/'configs')
    monkeypatch.setattr('etazero.config.ROOT',tmp_path)
    return tmp_path


@pytest.mark.parametrize('name',[
    'baseline','minimal_test','smoke_test','muzero','raw_muzero',
    'az_pcr/100v','az_pcr/200v','autoexp_example/arm_a',
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
    directory=version/'configs/az_pcr'
    plan=experiment_plan(directory,environ={},work_dir=version/'work')
    assert len(plan['arms'])==5
    for arm in plan['arms']:
        standalone=load_config(Path(arm['config_dir']))
        assert arm['run_dir']==str(version/'data/az_pcr'/arm['name'])
        assert arm['config']==standalone
        resolved=version/'generated'/arm['name']
        write_arm_config(arm['config'],resolved)
        assert load_config(resolved)==standalone
    assert not (version/'work').exists() and not (version/'data').exists()
